#include "so3_control/NetworkControl.h"
#include "ros/ros.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "network_ctrl_node");
    ros::NodeHandle node("~");
    NetworkControl controller(node);
    ros::spin();
    return 0;
}