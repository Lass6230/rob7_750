import rclpy
from rclpy.node import Node
import os
import sys

from std_msgs.msg import String
from .submodules import LB_optimizer_dyn as LB

from sensor_msgs.msg import LaserScan
import collections
import time

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import Twist
import tf2_py
from tf_transformations import euler_from_quaternion, quaternion_from_euler
import math
import numpy as np
import matplotlib.pyplot as plt
# from rclpy.qos_event import SubscriptionEventCallbacks
# from rclpy.qos_event import QoSSubscriptionEventType
import networkx as nx

class SafeRlNode(Node):

    def __init__(self):
        super().__init__('safe_rl_node')

        self.goal_counter = 0
        self.start_time = time.time()
        self.medium_room_goals_ = [[9.5,4.5],[9.5,-3.5],[0.0,-3.5],[9.5,3.5]]
        self.big_room_goals_ = [[7, 6 , 0.0],[11.0,0,0.0],[6.5,-7, 0.0],[2,0,0.0]]#[12.5, -5,0.0],[3.0,-6,0.0],[3.0,0.0,0.0],[12.5,6,0.0],[3.0,-5,0.0]]
        self.small_room_goals_ = [[],[]]
        buffer_size = 5
        self.cir_buffer_x_vel = collections.deque(maxlen=buffer_size)
        self.cir_buffer_y_vel = collections.deque(maxlen=buffer_size)
        self.cir_buffer_rot_vel = collections.deque(maxlen=buffer_size)
        self.trajectory_x = []
        self.trajectory_y = []
        self.whole_trajectory_x = []
        self.whole_trajectory_y = []
        self.amount_of_trajectories = 0
        self.vel_data = []
        self.fh0_data = []
        self.fh1_data = []
        self.xt_log_data = []
        self.elapsed_time_data = []
        self.whole_vel_data = []
        self.whole_fh0_data = []
        self.whole_fh1_data = []
        self.whole_xt_log_data = []
        self.amount_of_trajectories2 = 0
        

        self.cmd_vel_publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)

        grid_size = (100, 100)
        
       
        self.G = nx.grid_2d_graph(*grid_size)


        # self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds

        # self.timer = self.create_timer(timer_period, self.timer_callback)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.target_frame = "base_link"
        self.from_frame = "odom"
        
        
        self.actccepted_distance = 3
        self.safe_rl = LB.Simulation(ok_distance = self.actccepted_distance)
        #self.goal = [10.0,0.0, 0.0]
        #self.safe_rl.setGoal(self.goal[0],self.goal[1], self.goal[2])
        self.goal = self.big_room_goals_[self.goal_counter]
        self.goals = self.big_room_goals_
        #self.goal = [0.5,3.0, 0.0]
        self.safe_rl.setGoal(self.goal[0],self.goal[1], self.goal[2])

        qos_policy = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                                          history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                                          depth=1)
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.sensor_callback,
            qos_profile=qos_policy,
        )
        self.subscription  # prevent unused variable warning

        self.count = 0
        

        # self.cmd_vel_sub = self.create_subscription(Twist, "cmd_vel_test",self.cmd_vel_tester,10)
        # self.cmd_vel_sub

        # self.i = 0
        # sim = LB.Simulation()
    def cmd_vel_tester(self, msg):
        x,y,rot = self.location()
        x_vel = msg.linear.x*math.cos(-rot)-msg.linear.y*math.sin(-rot)
        y_vel = msg.linear.y*math.cos(-rot)+msg.linear.x*math.sin(-rot)
        rot_vel = msg.angular.z
        self.publish_cmd_vel(x_vel,y_vel,rot_vel)

    def sensor_callback(self, msg):
        # print("lidar points: ", len(msg.ranges))
        x,y,rot = self.location()
        
        # self.get_logger().info('Number of points: "%i"' % len(msg.ranges))
        obstacles_x = []
        obstacles_y = []
        obstacles = [()]
        
        # obstacles_x.append(math.cos(msg.angle_min+(1*msg.angle_increment))*msg.ranges[0])
        # self.get_logger().info('X: "%f"' % obstacles_x[0])
        # self.get_logger().info('X: "%f" ' % x)
        # self.get_logger().info('Y: "%f"' % y)
        for i in range(len(msg.ranges)):
            
                
            
            obstacles_x.append((math.cos(msg.angle_min+(i*msg.angle_increment)+rot)*msg.ranges[i])+x)
            obstacles_y.append((math.sin(msg.angle_min+(i*msg.angle_increment)+rot)*msg.ranges[i])+y)
            obstacles.append([((math.cos(msg.angle_min+(i*msg.angle_increment)+rot)*msg.ranges[i])+x), ((math.sin(msg.angle_min+(i*msg.angle_increment)+rot)*msg.ranges[i])+y)]) # mabye adding the offset from baselink
            # mabye adding the offset from baselink
            
        flattened_obs_zone_1 = [item for item in obstacles[0:int(len(obstacles)/3)] if isinstance(item, list) and item != [float('inf'), float('inf')] and item !=[float('-inf'), float('-inf')]and item != [float('-inf'), float('inf')]and item != [float('inf'), float('-inf')]]
        flattened_obs_zone_2 = [item for item in obstacles[int(len(obstacles)/3):int((len(obstacles)/3)*2)] if isinstance(item, list) and item != [float('inf'), float('inf')] and item !=[float('-inf'), float('-inf')]and item != [float('-inf'), float('inf')]and item != [float('inf'), float('-inf')]]
        flattened_obs_zone_3 = [item for item in obstacles[int((len(obstacles)/3)*2):int(len(obstacles))] if isinstance(item, list) and item != [float('inf'), float('inf')] and item !=[float('-inf'), float('-inf')]and item != [float('-inf'), float('inf')]and item != [float('inf'), float('-inf')]]
        flattened_obs = [item for item in obstacles if isinstance(item, list) and item != [float('inf'), float('inf')] and item !=[float('-inf'), float('-inf')]and item != [float('-inf'), float('inf')]and item != [float('inf'), float('-inf')]]
        
        if self.goalChecker(x,y):
            self.update_plot(x,y)
            self.publish_cmd_vel(0.0,0.0,0.0)
            self.get_logger().info('Goal Reached')
            self.goal_counter += 1
            if self.goal_counter < len(self.goals):
                 self.goal = self.goals[self.goal_counter]
                 self.safe_rl.setGoal(self.goal[0],self.goal[1], self.goal[2])
            #self.goal = [np.random.random_integers(0,6), np.random.random_integers(-3,1), np.random.random_integers(-3.14,3.14)]
            # self.safe_rl.setGoal(self.goal[0],self.goal[1], self.goal[2])
            self.cir_buffer_x_vel.clear()
            self.cir_buffer_y_vel.clear()
            self.cir_buffer_rot_vel.clear()
        else:
            self.safe_rl.setPos(x=x,y=y,rot=rot)
            self.safe_rl.update()
            vel = self.safe_rl.getCmdVel()
            
            xt_log = self.safe_rl.getXtLog()
            fh = self.safe_rl.getFandH()
            
            elapsed_time = time.time() - self.start_time
            self.xt_log_data.append(xt_log)
            self.vel_data.append(vel)  # Assuming vel is a list
            
            self.fh0_data.append(fh[0])
            self.fh1_data.append(fh[1])
    
            self.elapsed_time_data.append(elapsed_time)

            # Update plot
            self.update_plot(x,y)

            

            
            x_vel = vel[0]*math.cos(-rot)-vel[1]*math.sin(-rot)
            y_vel = vel[1]*math.cos(-rot)+vel[0]*math.sin(-rot)
            # x_vel = vel[0]
            # y_vel = vel[1]
            rot_vel = vel[2]
            if x_vel > 0.9:
                x_vel = 0.9
            if y_vel > 0.9:
                y_vel = 0.9
            if rot_vel > 0.9:
                rot_vel = 0.9
            if x_vel < -0.9:
                x_vel = -0.9
            if y_vel < -0.9:
                y_vel = -0.9
            if rot_vel < -0.9:
                rot_vel = -0.9



            self.cir_buffer_x_vel.append(x_vel)
            self.cir_buffer_y_vel.append(y_vel)
            self.cir_buffer_rot_vel.append(rot_vel)
            x_vel= (sum(self.cir_buffer_x_vel)/len(self.cir_buffer_x_vel))
            y_vel= (sum(self.cir_buffer_y_vel)/len(self.cir_buffer_y_vel))
            rot_vel= (sum(self.cir_buffer_rot_vel)/len(self.cir_buffer_rot_vel))
            self.publish_cmd_vel(x_vel,y_vel,rot_vel)
        
        close, cl1, cl2, cl3 = self.safe_rl.closest_arrays_to_zero(flattened_obs, 25,x,y, flattened_obs_zone_1 ,flattened_obs_zone_2, flattened_obs_zone_3)

        # print("WOW THATS ALOT OF ARRAY",close)

        # remember that plotting is to slow, we will then miss laserscan

        # self.ax.clear()
        # self.ax.scatter(y,x,color='red')
        # self.ax.scatter(obstacles_y,obstacles_x)
        
        # Plot each point from the 'close' arrays

        # for point in cl1:
        #     self.ax.scatter(point[1], point[0], color='magenta')  # Assuming each point in 'close' is [y, x]
        # for point in cl2:
        #     self.ax.scatter(point[1], point[0], color='cyan')  # Assuming each point in 'close' is [y, x]
        # for point in cl3:
        #     self.ax.scatter(point[1], point[0], color='yellow')  # Assuming each point in 'close' is [y, x]
        # self.ax.scatter(self.goal[1], self.goal[0], color='green')  # Assuming each point in 'close' is [y, x]    

        # plt.pause(0.005)

            
            # self.publish_cmd_vel(vel[0],vel[1],vel[2])
       

        # self.ax.clear()
        # self.ax.scatter(y,x,color='red')
        # self.ax.scatter(obstacles_y,obstacles_x)
        # plt.pause(0.005)


    def update_plot(self,x,y):
            # Clear the plot for redrawing
           

            # Plot elapsed time vs vel
            fig1, ax1 = plt.subplots()
            ax1.plot(self.elapsed_time_data, [vel_tuple[0] for vel_tuple in self.vel_data], color='blue', label='x-velocity')
            ax1.plot(self.elapsed_time_data, [vel_tuple[1] for vel_tuple in self.vel_data], color='red', label='y-velocity')
            ax1.plot(self.elapsed_time_data, [vel_tuple[2] for vel_tuple in self.vel_data], color='green', label='rot-velocity')
            ax1.set_xlabel('Elapsed Time (s)')
            ax1.set_ylabel('Velocity')
            ax1.set_title('Velocity vs Elapsed Time')
            if not ax1.get_legend():
                ax1.legend()

            # Plot elapsed time vs fh0
            fig2, ax2 = plt.subplots()
            ax2.plot(self.elapsed_time_data, self.fh0_data, color='blue', label='f-value')
            ax2.set_xlabel('Elapsed Time (s)')
            ax2.set_ylabel('f-value')
            ax2.set_title('f-value vs Elapsed Time')
            if not ax2.get_legend():
                ax2.legend()

            # Plot elapsed time vs fh1
            fig3, ax3 = plt.subplots()
            ax3.plot(self.elapsed_time_data, self.fh1_data, color='green', label='h-value')
            ax3.set_xlabel('Elapsed Time (s)')
            ax3.set_ylabel('h-value')
            ax3.set_title('h-value vs Elapsed Time')
            if not ax3.get_legend():
                ax3.legend()

            # Plot elapsed time vs xt_log
            fig4, ax4 = plt.subplots()
            ax4.plot(self.elapsed_time_data, [xt_data[0] for xt_data in self.xt_log_data], color='blue', label='next x value')
            ax4.plot(self.elapsed_time_data, [xt_data[1] for xt_data in self.xt_log_data], color='red', label='next y value')
            ax4.plot(self.elapsed_time_data, [xt_data[2] for xt_data in self.xt_log_data], color='green', label='next rot value')
            ax4.set_xlabel('Elapsed Time (s)')
            ax4.set_ylabel('xt/next location')
            ax4.set_title('xt vs Elapsed Time')
            if not ax4.get_legend():
                ax4.legend()

            # Specify the directory to save the plots
            save_dir = '/home/robotlab/p7_ws/src/Plots'  # Change this to the desired directory
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            distance = math.sqrt(pow(self.goal[0] - x, 2) + pow(self.goal[1] - y, 2))
            if distance < self.actccepted_distance:
                fig1.savefig(os.path.join(save_dir, f'velocity_plot_{self.goal_counter}.png'))
                fig2.savefig(os.path.join(save_dir, f'f_plot_{self.goal_counter}.png'))
                fig3.savefig(os.path.join(save_dir, f'h-plot_{self.goal_counter}.png'))
                fig4.savefig(os.path.join(save_dir, f'xt_log_plot_{self.goal_counter}.png'))

            # Close the figures to release resources
            plt.close(fig1)
            plt.close(fig2)
            plt.close(fig3)
            plt.close(fig4)

            # Pause for a short time to avoid overwhelming the plot
            plt.pause(0.05)


    def location(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.from_frame,
                self.target_frame,
                rclpy.time.Time())

            orientation_list = [
                t.transform.rotation.x,
                t.transform.rotation.y,
                t.transform.rotation.z,
                t.transform.rotation.w
            ]

            (roll, pitch, yaw) = euler_from_quaternion(orientation_list)

            self.pose_yaw = yaw

            x, y = t.transform.translation.x, t.transform.translation.y

            self.trajectory_x.append(x)
            self.trajectory_y.append(y)
            self.whole_trajectory_x.append(x)
            self.whole_trajectory_y.append(y)

            return x, y, yaw
        except TransformException as ex:
            return 0.0, 0.0, 0.0
    
    

        
    def publish_cmd_vel(self,x_vel,y_vel,z_rot_vel):
        data = Twist()
        data.linear.x = x_vel
        data.linear.y = y_vel
        
        data.angular.z = z_rot_vel
        self.cmd_vel_publisher_.publish(data)
    
    def goalChecker(self, x, y):
        distance = math.sqrt(pow(self.goal[0] - x, 2) + pow(self.goal[1] - y, 2))
        fig5, ax5 = plt.subplots()

        # Plot the trajectory with red dot at the current robot position
        ax5.plot(self.trajectory_x, self.trajectory_y, label='Robot Trajectory')
        ax5.scatter(self.trajectory_x[0], self.trajectory_y[0], color='green', label='Start Position')
        ax5.scatter(x, y, color='red', label='Goal position')
        ax5.legend()
        ax5.set_xlabel('X')
        ax5.set_ylabel('Y')
        ax5.set_title('Robot Trajectory')

        
        if distance < self.actccepted_distance:
            # Create a new figure and axis for plotting
            
            
            # Specify the directory to save the plot
            save_dir = '/home/robotlab/p7_ws/src/Experiments_trajectory'  # Change this to the desired directory
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Save the plot as an image file
            save_path = os.path.join(save_dir, f'robot_trajectory_goal_{self.goal_counter}.png')
            plt.savefig(save_path)

            # Clear the plot for the next iteration
            

            plt.close(fig5)
            if self.amount_of_trajectories < len(self.goals):
                # Update the goal and clear the trajectory
                self.amount_of_trajectories += 1
                self.goal = self.goals[self.goal_counter]
                self.safe_rl.setGoal(self.goal[0], self.goal[1], self.goal[2])
                self.trajectory_x.clear()
                self.trajectory_y.clear()
                

                print("this is what goal we are at", self.goal_counter)

            if self.amount_of_trajectories == len(self.goals):
                
                # Create a new figure and axis for plotting
                fig, ax = plt.subplots()

                ax.plot(self.whole_trajectory_x, self.whole_trajectory_y, label='Robot Trajectory')
                ax.scatter(self.whole_trajectory_x[0], self.whole_trajectory_y[0], color='green', label='Start Position')
                ax.scatter(x, y, color='red', label='Goal position')
                ax.legend()
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title('Robot Trajectory')

                # Save the plot as an image file
                save_path = os.path.join(save_dir, f'whole_robot_trajectory_goal.png')
                plt.savefig(save_path)

                # Clear the plot for the next iteration
                plt.close(fig)

            return True
        else:
            plt.close(fig5)
            return False
            
        
def main(args=None):
    rclpy.init(args=args)

    safe_rl_node = SafeRlNode()

    rclpy.spin(safe_rl_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    safe_rl_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()