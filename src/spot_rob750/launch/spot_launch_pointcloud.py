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
	ld.add_action(DeclareLaunchArgument('use_sim_time', default_value = "False", description='to change when using simulations or bag files'))



	# Nav2
	

	# Spot driver
	if 1:
		spot_config = os.path.join(get_package_share_directory('spot_rob750'), 'config/spot_config.yaml')
		spot_launch = os.path.join(get_package_share_directory('spot_driver'), 'launch', "spot_driver.launch.py")
		spot_pointcloud_launch = os.path.join(get_package_share_directory('spot_driver'), 'launch', "point_cloud_xyz.launch.py")
		ld.add_action(IncludeLaunchDescription(PythonLaunchDescriptionSource(spot_launch),
		                                       launch_arguments={"config_file": spot_config}.items()))

		ld.add_action(IncludeLaunchDescription(PythonLaunchDescriptionSource(spot_pointcloud_launch),
		                                       launch_arguments={"config_file": spot_config}.items()))
		# remappings = [
		# 	('imu', '/my/data'),
		# 	('rgb/image', '/camera/color/image_raw'),  # "/camera/infra1/image_rect_raw"),  #
		# 	('rgb/camera_info', '/camera/color/camera_info'), # "/camera/infra1/camera_info"),  #
		# 	('depth/image', '/camera/aligned_depth_to_color/image_raw'),
		# 	('depth/image_info', '/camera/aligned_depth_to_color/image_raw_info')

		# ]

		
	if 1:
		ld.add_action(Node(
				package='rviz2',
				namespace='rviz2',
				executable='rviz2',
				name='rviz2',
				arguments=["-d", os.path.join(get_package_share_directory("spot_rob750"), "config", "spot_rviz2_config.rviz")],
				output='log'
		))

	
	"---------------------------------------------------------------"
	
	return ld