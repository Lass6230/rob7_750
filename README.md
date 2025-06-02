# Rob7_750
Rob 7 group 750 semester project

## Install
1.     git clone https://github.com/Lass6230/rob7_750.git
    
2.     pip3 install bosdyn-client bosdyn-mission bosdyn-api bosdyn-core
3.     pip install networkx cvxopt numdifftools transforms3d
4.     sudo apt install ros-humble-joint-state-publisher-gui ros-humble-xacro
5.     sudo apt install ros-humble-tf-transformations
6.     cd rob7_750/src
7.     git clone -b humble https://github.com/Lass6230/rob7_750.git
8.     cd ..
9.     colcon build --symlink-install
    

## In spot_driver spot_login.yaml
    
    hostname: "192.168.80.3"
    cmd_duration: 0.125

## In pot_driver spot_ros.yaml
    rates:
      robot_state: 20.0
      metrics: 0.04
      lease: 1.0
      front_image: 10.0
      side_image: 10.0
      rear_image: 10.0
    auto_claim: True
    auto_power_on: True
    auto_stand: True
    deadzone: 0.05
    estop_timeout: 9.0
## Launch Driver
    ros2 launch spot_rob750 spot_launch.py


## Example on service command
    ros2 service call /stand std_srvs/srv/Trigger
    ros2 service call /sit std_srvs/srv/Trigger
    ros2 service call /power_on std_srvs/srv/Trigger
    ros2 service call /power_off std_srvs/srv/Trigger
    ros2 service call /estop/gentle std_srvs/srv/Trigger
    ros2 service call /estop/release std_srvs/srv/Trigger

## Dont use estop hard



sudo apt install ros-humble-pointcloud-to-laserscan



Spot_driver: https://github.com/bdaiinstitute/spot_ros2
