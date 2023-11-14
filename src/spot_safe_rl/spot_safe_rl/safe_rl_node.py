import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from .submodules import LB_optimizer as LB



from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import Twist

class SafeRlNode(Node):

    def __init__(self):
        super().__init__('safe_rl_node')
        # self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds

        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.target_frame = "base_link"
        self.from_frame = "map"
        # self.i = 0
        # sim = LB.Simulation()


    def timer_callback(self):
        # msg = String()
        # msg.data = 'Hello World: %d' % self.i
        # # self.publisher_.publish(msg)
        # self.get_logger().info('not Publishing: "%s"' % msg.data)
        # self.i += 1

        try:
            t = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.from_frame,
                rclpy.time.Time())
            self.get_logger().info(
                        f'got transform {self.target_frame} to {self.target_frame}')
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {self.target_frame} to {self.target_frame}: {ex}')
            return
    
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