# UAV Simulator

### Build:
```bash
catkin_make
```

### MODE 1. PID Position Controller and Simulator
Work with traditional planner
```
source devel/setup.bash
roslaunch so3_quadrotor_simulator simulator_position_control.launch
```

### MODE 2. Attitude Controller with Disturbance Observer
Work with our learning-based planner (without position controller)
```
source devel/setup.bash
roslaunch so3_quadrotor_simulator simulator_attitude_control.launch
```

### Others
pub disturbance
```
rostopic pub /force_disturbance
```

takeoff and land (used in realworld flight)
```
rosservice call /network_controller_node/takeoff_land
```

### acknowledgment

This repo is modified from https://github.com/HKUST-Aerial-Robotics/Fast-Planner, thanks for their excellent work!