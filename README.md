# Rob7_750
Rob 7 group 750 semester project

## Install
    git clone https://github.com/Lass6230/rob7_750.git
    
    pip3 install bosdyn-client bosdyn-mission bosdyn-api bosdyn-core
    sudo apt install ros-humble-joint-state-publisher-gui ros-humble-xacro
    cd rob7_750/src
    git clone -b humble https://github.com/MASKOR/Spot-ROS2.git src/
    colcon build --symlink-install

## In spot_login.yaml
    username: "admin"
    password: "mt9pe6pa0rm5"
    hostname: "192.168.80.3"
    cmd_duration: 0.125



Spot_driver: https://github.com/bdaiinstitute/spot_ros2
