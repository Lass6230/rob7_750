import rclpy
from rclpy.node import Node

from std_msgs.msg import String
# from .submodules import LB_optimizer as LB

from sensor_msgs.msg import LaserScan

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import Twist
import tf2_py
from tf_transformations import euler_from_quaternion
import math
import numpy as np
import matplotlib.pyplot as plt
# from rclpy.qos_event import SubscriptionEventCallbacks
# from rclpy.qos_event import QoSSubscriptionEventType
import networkx as nx

class SafeRlNode(Node):

    def __init__(self):
        super().__init__('safe_rl_node')



        grid_size = (100, 100)
        
        self.fig, self.ax = plt.subplots()
        self.G = nx.grid_2d_graph(*grid_size)


        # self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds

        # self.timer = self.create_timer(timer_period, self.timer_callback)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.target_frame = "base_link"
        self.from_frame = "odom"

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

        # self.i = 0
        # sim = LB.Simulation()

    def sensor_callback(self, msg):
        # print("sensor callback")

        x,y,rot = self.location()
        self.get_logger().info('Number of points: "%i"' % len(msg.ranges))
        obstacles_x = []
        obstacles_y = []
        # obstacles_x.append(math.cos(msg.angle_min+(1*msg.angle_increment))*msg.ranges[0])
        # self.get_logger().info('X: "%f"' % obstacles_x[0])
        self.get_logger().info('X: "%f" ' % x)
        self.get_logger().info('Y: "%f"' % y)
        for i in range(len(msg.ranges)):
            # obstacles_x.append((math.cos(msg.angle_min+(i*msg.angle_increment))*msg.ranges[i])*(math.cos(rot)-math.sin(rot))-x)
            # obstacles_y.append((math.sin(msg.angle_min+(i*msg.angle_increment))*msg.ranges[i])*(math.sin(rot)-math.cos(rot))+y)
            obstacles_x.append((math.cos(msg.angle_min+(i*msg.angle_increment)+rot)*msg.ranges[i])+x)
            obstacles_y.append((math.sin(msg.angle_min+(i*msg.angle_increment)+rot)*msg.ranges[i])+y)
            # if msg.angle_min+(i*msg.angle_increment) < 0.0:
            #     obstacles_x.append(-math.cos(msg.angle_min+(i*msg.angle_increment))*msg.ranges[i])
            #     obstacles_y.append(-math.sin(msg.angle_min+(i*msg.angle_increment))*msg.ranges[i])
            # else:
            #     obstacles_x.append(math.cos(msg.angle_min+(i*msg.angle_increment))*msg.ranges[i])
            #     obstacles_y.append(math.sin(msg.angle_min+(i*msg.angle_increment))*msg.ranges[i])
            # self.get_logger().info('X: "%f" ' % obstacles_x[i])
            # self.get_logger().info('Y: "%f"' % obstacles_y[i])
        self.ax.clear()
        self.ax.scatter(y,x,color='red')
        self.ax.scatter(obstacles_y,obstacles_x)
        plt.pause(0.005)
        # self.timer_callback()

    def location(self):
        # msg = String()
        # msg.data = 'Hello World: %d' % self.i
        # # self.publisher_.publish(msg)
        # self.get_logger().info('not Publishing: "%s"' % msg.data)
        # self.i += 1

        try:
            t = self.tf_buffer.lookup_transform(
                self.from_frame,
                self.target_frame,
                rclpy.time.Time())
            self.get_logger().info(
                        f'got transform {self.target_frame} to {self.target_frame}')
            self.get_logger().info('x: "%f"' % t.transform.translation.x)
            self.get_logger().info('y: "%f"' % t.transform.translation.y)
            orientation_list = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]

            (roll, pitch, yaw) = euler_from_quaternion(orientation_list)

            self.pose_yaw = yaw

            self.get_logger().info('z_rot: "%f"' % yaw)
            return t.transform.translation.x, t.transform.translation.y, yaw
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {self.target_frame} to {self.target_frame}: {ex}')
            return 0.0, 0.0, 0.0
    
    def get_robot_position(self):
        print("getting robot position")

        
    def publish_cmd_vel(self):
        print("publishing cmd_vel")

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