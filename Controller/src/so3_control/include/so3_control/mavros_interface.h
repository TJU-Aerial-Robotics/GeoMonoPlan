#ifndef MAVROS_INTERFACE_H_
#define MAVROS_INTERFACE_H_

#include "ros/ros.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>

#include <mavros_msgs/AttitudeTarget.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/CommandBool.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

class Mavros_Interface
{
public:
    Mavros_Interface() : _nh("~cmd")
    {
        _state.reset();
        std::string base_name = "/mavros";
        // char id_str[10];
        // sprintf(id_str, "%d", id);
        // base_name += id_str;

        std::string att_target_pub_name;
        att_target_pub_name = base_name + "/setpoint_raw/attitude";
        att_target_pub = _nh.advertise<mavros_msgs::AttitudeTarget>(att_target_pub_name.c_str(), 10);

        std::string state_sub_name;
        state_sub_name = base_name + "/state";
        state_sub = _nh.subscribe(state_sub_name.c_str(), 10, &Mavros_Interface::state_cb, this);

        std::string set_mode_s_name;
        set_mode_s_name = base_name + "/set_mode";
        set_mode_client = _nh.serviceClient<mavros_msgs::SetMode>(set_mode_s_name.c_str());

        std::string arm_disarm_s_name;
        arm_disarm_s_name = base_name + "/cmd/arming";
        arm_disarm_client = _nh.serviceClient<mavros_msgs::CommandBool>(arm_disarm_s_name.c_str());
    }

    ~Mavros_Interface() {}

    typedef struct mavros_state_t
    {
        ros::Time header;
        bool has_armed;
        bool offboard_enabled;
        void reset()
        {
            has_armed = false;
            offboard_enabled = false;
        }
        mavros_state_t()
        {
            reset();
        }
    };

    void state_cb(const mavros_msgs::State &state_data)
    {
        mavros_msgs::State temp_data = state_data;
        _state.header = state_data.header.stamp;
        _state.has_armed = state_data.armed;
        if (state_data.mode == "OFFBOARD")
        {
            _state.offboard_enabled = true;
        }
        else
        {
            _state.offboard_enabled = false;
        }
    }

    void get_status(bool &arm_state, bool &offboard_enabled)
    {
        arm_state = _state.has_armed;
        offboard_enabled = _state.offboard_enabled;
    }

    bool set_arm_and_offboard()
    {
        ros::Rate _ofb_check_rate(1);
        int try_arm_ofb_times = 0;
        while (!_state.offboard_enabled || !_state.has_armed)
        {
            if (_state.offboard_enabled)
            {
                ros::Rate _arm_check_rate(1);
                while (!_state.has_armed)
                {
                    mavros_msgs::CommandBool arm_srv;
                    arm_srv.request.value = true;
                    if (arm_disarm_client.call(arm_srv))
                    {
                        ROS_INFO("vehicle ARMED");
                    }
                    try_arm_ofb_times = try_arm_ofb_times + 1;
                    if (try_arm_ofb_times >= 3)
                    {
                        ROS_ERROR("try 3 times, cannot armed uav, give up!");
                        return false;
                    }
                    _arm_check_rate.sleep();
                }
            }
            else
            {
                ROS_INFO("not in OFFBOARD mode");
                mavros_msgs::SetMode set_mode_srv;
                set_mode_srv.request.base_mode = 0;
                set_mode_srv.request.custom_mode = "OFFBOARD";
                if (!set_mode_client.call(set_mode_srv))
                {
                    return false;
                }
                ROS_INFO("switch to OFFBOARD mode");
                _ofb_check_rate.sleep();
            }
        }
        return true;
    }

    /* void set_arm(const ros::TimerEvent& event) {
        if(_state.offboard_enabled) {
            mavros_msgs::CommandBool arm_srv;
            arm_srv.request.value = true;
            if (arm_disarm_client.call(arm_srv)) {
                ROS_INFO("vehicle ARMED");
            }
        } else {
            ROS_INFO("not in OFFBOARD mode");
        }
    } */

    bool set_disarm()
    {
        ros::Rate _arm_check_rate(1);
        while (_state.has_armed)
        {
            mavros_msgs::CommandBool arm_srv;
            arm_srv.request.value = false;
            if (!arm_disarm_client.call(arm_srv))
            {
                return false;
            }
            ROS_INFO("vehicle DISARMED");
            _arm_check_rate.sleep();
        }
        return true;
    }

    void pub_att_thrust_cmd(const Eigen::Quaterniond &q_d, const double &thrust_d)
    {
        /*
            Convert orientation to the MAVROS NED convention here to preserve the PID interface.
        */
        mavros_msgs::AttitudeTarget at_cmd;
        at_cmd.header.stamp = ros::Time::now();
        at_cmd.type_mask = at_cmd.IGNORE_ROLL_RATE | at_cmd.IGNORE_PITCH_RATE | at_cmd.IGNORE_YAW_RATE;
        at_cmd.thrust = (float)thrust_d;
        at_cmd.orientation.w = q_d.w();
        at_cmd.orientation.x = q_d.x();
        at_cmd.orientation.y = -q_d.y();
        at_cmd.orientation.z = -q_d.z();
        att_target_pub.publish(at_cmd);
    }

    // for simulation
    void set_arm_and_offboard_manually()
    {
        _state.has_armed = true;
        _state.offboard_enabled = true;
    }

    void set_disarm_manually()
    {
        _state.has_armed = false;
    }

private:
    ros::NodeHandle _nh;
    ros::Publisher att_target_pub;
    ros::Subscriber state_sub;
    ros::ServiceClient set_mode_client;
    ros::ServiceClient arm_disarm_client;
    mavros_state_t _state;
};

#endif
