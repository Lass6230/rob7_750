# Rob7_750
Rob 7 group 750 semester project

## Install
    git clone https://github.com/Lass6230/rob7_750.git
    
    pip3 install bosdyn-client bosdyn-mission bosdyn-api bosdyn-core
    sudo apt install ros-humble-joint-state-publisher-gui ros-humble-xacro
    cd rob7_750/src
    git clone -b humble https://github.com/MASKOR/Spot-ROS2.git src/
    colcon build --symlink-install

## In spot_driver spot_login.yaml
    username: "admin"
    password: "mt9pe6pa0rm5"
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

Spot_driver: https://github.com/bdaiinstitute/spot_ros2
