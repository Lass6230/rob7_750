# # Import necessary ROS2 packages
# import os
# from ament_index_python.packages import get_package_share_directory
# from launch import LaunchDescription, actions
# from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
# from launch.launch_description_sources import PythonLaunchDescriptionSource
# from launch.substitutions import LaunchConfiguration, Command
# from launch_ros.actions import Node
# import launch_ros
# from math import pi


# def generate_launch_description():
# 	ld = LaunchDescription()

# 	ld.add_action(DeclareLaunchArgument(
#         "sim_mode", default_value="False", description="sim or not"
#     	))
# 	sim_mode = LaunchConfiguration("sim_mode", default='False')
	
# 	if sim_mode:
# 		ld.add_action(Node(
# 				package="spot_safe_rl",
# 				executable="safe_rl_node",
# 				name="safe_rl_node",
# 				output="screen",
# 				))
# 	else:
# 		remappings = [
# 				('scan','sick_tim_5xx/scan'),
# 			]
# 		# Static TF
# 		ld.add_action(Node(
# 				package="spot_safe_rl",
# 				executable="safe_rl_node",
# 				name="safe_rl_node",
# 				output="screen",
# 				remappings=remappings,
# 				))
	
# 	return ld


import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():


	sim_mode_arg = DeclareLaunchArgument(
        "sim_mode", default_value="False", description="communication"
    )
    
	com = LaunchConfiguration("com", default="True", )

	sim_mode = LaunchConfiguration("sim_mode", default='True')
	
	remappings = [
		('scan','sick_tim_5xx/scan'),
	]
	# if sim_mode:
	
	# Static TF
	safe_rl = Node(
			package="spot_safe_rl",
			executable="safe_rl_node_dyn",
			name="safe_rl_node_dyn",
			output="screen",
			remappings=remappings,
			)
	
	safe_rl2 = Node(
			package="spot_safe_rl",
			executable="safe_rl_node_dyn",
			name="safe_rl_node_dyn",
			output="screen",
			)
   
	return LaunchDescription(
		[   
            # gripper_launch,
            # safe_rl,
			safe_rl2,
            sim_mode_arg,
            
		]
    )