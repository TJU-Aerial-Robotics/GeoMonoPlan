# Controller workspace

This ROS1 catkin workspace provides the controller, quadrotor simulator, and
`quadrotor_msgs/PositionCommand` used by the planner. Only source files are
distributed. Build the workspace locally before starting ROS inference.

## Build locally

Use a terminal with ROS1 Noetic and its system Python environment available.
The workspace requires Eigen3, Armadillo, and the ROS dependencies declared in
the package manifests.

From the GeoMonoPlan repository root:

```bash
source /opt/ros/noetic/setup.bash
cd Controller
rosdep install --from-paths src --ignore-src -r -y
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash
```

The generated `build/`, `devel/`, and `install/` directories are not distributed
or committed. Source `Controller/devel/setup.bash` in each terminal that runs
the planner or controller.

## Simulation

For the acceleration-command controller used by the learned planner:

```bash
roslaunch so3_quadrotor_simulator simulator_attitude_control.launch
```

For the PID position-controller alternative:

```bash
roslaunch so3_quadrotor_simulator simulator_position_control.launch
```

The simulation publishes `/sim/odom` and accepts planner reference commands on
`/so3_control/pos_cmd`. Configure the external image source to publish
`/rgb/image` before running the planner.

## Acknowledgment

The original controller workspace acknowledges
[Fast-Planner](https://github.com/HKUST-Aerial-Robotics/Fast-Planner).
Existing upstream copyright and license notices are preserved in the source.
