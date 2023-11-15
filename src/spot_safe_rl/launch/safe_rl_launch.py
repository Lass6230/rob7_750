# Import necessary ROS2 packages
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, actions
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
import launch_ros
from math import pi


def generate_launch_description():
	ld = LaunchDescription()

	remappings = [
			('scan','sick_tim_5xx/scan'),
	]
	# Static TF
	ld.add_action(Node(
			package="spot_safe_rl",
			executable="safe_rl_node",
			name="safe_rl_node",
			output="screen",
			remappings=remappings,
		    ))
	
	return ld