import mediapipe as mp
from mediapipe_classes import MediaPipePose
import numpy as np
import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from utbots_actions.msg import MPPoseAction, MPPoseResult
from sensor_msgs.msg import Image
from vision_msgs.msg import Skeleton2d
from geometry_msgs.msg import PointStamped, Point, TransformStamped
from std_msgs.msg import String, Bool
from tf2_ros import TransformBroadcaster

class MediaPipeNode(Node, MediaPipePose):
    def __init__(self):
        Node.__init__(self, 'mediapipe_node')
        MediaPipePose.__init__(self)

        # Set parameters
        self.declare_parameter('topic_namespace', "/utbots/vision")
        self.declare_parameter('num_poses', 1)
        self.declare_parameter('detection_conf', 0.5)
        self.declare_parameter('presence_conf', 0.5)
        self.declare_parameter('track_conf', 0.5)
        self.declare_parameter('segmentation_mask', False)
        self.declare_parameter('rbg_topic', "camera/rgb/image_color")

        self.topic_namespace = self.get_parameter('topic_namespace').get_parameter_value().string_value
        self.num_poses = self.get_parameter('num_poses').get_parameter_value().integer_value
        self.detection_conf = self.get_parameter('detection_conf').get_parameter_value().double_value
        self.presence_conf = self.get_parameter('presence_conf').get_parameter_value().double_value
        self.track_conf = self.get_parameter('track_conf').get_parameter_value().double_value
        self.segmentation_mask = self.get_parameter('segmentation_mask').get_parameter_value().bool_value
        self.rgb_topic = self.get_parameter('rbg_topic').get_parameter_value().string_value

        # Define ROS messages
        self.msg_pose_landmarks = Skeleton2d()
        self.msg_target_skeleton_img = Image()
        self.msg_target_point = PointStamped()

        # Publishers and Subscribers
        self.sub_rgbImg = self.create_subscription(
            Image,
            self.rgb_topic,
            self.callback_rgb_img,
            10
        )
        self.pub_target_skeletonImg = self.create_publisher(Image, "pose/skeleton_img", 10)
        self.pub_poseLandmarks = self.create_publisher(Skeleton2d, "pose/pose_landmarks", 10)
        self.pub_target_point = self.create_publisher(PointStamped, "pose/torso_point", 10)
        self.pub_tf = TransformBroadcaster(self)

        # OpenCV image format conversion
        self.cvBridge = CvBridge()
        self.cv_img = None

        # Service to enable/disable synchronous processing
        self.srv_enable_sync = self.create_service(
            Bool,
            'enable_synchronous',
            self.enable_synchronous_callback
        )
        self.enable_synchronous = False

        # Action server initialization
        self._as = ActionServer(self, MPPoseAction, 'mediapipe_pose', self.pose_action_callback)
        
        self.synchronous_loop()

    # Callbacks
    def callback_rgb_img(self, msg):
        self.msg_rgb_img = msg

    def enable_synchronous_callback(self, request, response):
        self.enable_synchronous = request.data
        response.success = True
        response.message = f"Synchronous processing {'enabled' if self.enable_synchronous else 'disabled'}."
        return response

    def pose_action_callback(self, goal):
        self.get_logger().info("[MPPOSE] Goal received")

        # Optional specific image sent as goal
        if goal.Image.width != 0 and goal.Image.height != 0:
            img_msg = goal.Image
        else:
            img_msg = self.msg_rgb_img

        # Manage action results
        action_res = MPPoseResult()
        action_res.Success = Bool()

        pose_landmarks, skeleton_img, target_point = self.preprocess_and_predict(img_msg, draw=goal.GetDrawn, torso_point=goal.GetTorsoPoint)

        if pose_landmarks:
            self.pub_poseLandmarks.publish(pose_landmarks)
            
            if goal.GetTorsoPoint:
                action_res.Point(target_point)
                self.pub_target_point.publish(target_point)

            if goal.GetDrawn:
                self.pub_target_skeletonImg(skeleton_img)
            
            action_res.Success = True
            self._as.set_succeeded(action_res)
        else:
            action_res.Success = False
            self._as.set_aborted()

    # Methods
    def synchronous_loop(self):
        if self.enable_synchronous:
            img_msg = self.msg_rgb_img
            pose_landmarks, _, _ = self.preprocess_and_predict(img_msg, draw=False, torso_point=False)
            self.pub_poseLandmarks.publish(pose_landmarks)

    def preprocess_and_predict(self, img_msg, draw=False, torso_point=False):
        cv_img = self.imgmsg_to_cv(img_msg)
        pose_landmarks, drawn_img = self.predict_landmarks(cv_img, draw=draw)
                
        self.msg_pose_landmarks.points = pose_landmarks
        self.msg_target_skeleton_img = self.cvBridge.cv2_to_imgmsg(drawn_img)
        
        if torso_point:
            x, y, z = self.calculate_representative_point(pose_landmarks)
            self.msg_target_point.point = Point(x, y, z)
            self.msg_target_point.header.stamp = self.get_clock().now().to_msg()
            self.msg_target_point.header.frame_id = "camera_link"

            # Set the translations and rotations
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = "camera_link"
            transform.child_frame_id = "torso_point"
            transform.transform.translation.x = x
            transform.transform.translation.y = y
            transform.transform.translation.z = z
            transform.transform.rotation.x = 0.0
            transform.transform.rotation.y = 0.0
            transform.transform.rotation.z = 0.0
            transform.transform.rotation.w = 1.0
            self.pub_tf.sendTransform(transform)

        return self.msg_pose_landmarks, self.msg_target_skeleton_img, self.msg_target_point        

    def imgmsg_to_cv(self, img_msg):
        # Check if the image message is available
        if img_msg is not None:
            try:
                # Convert the ROS image message to an OpenCV image
                cv_img = self.cvBridge.imgmsg_to_cv2(img_msg, "rbg8")
                return cv_img

            except Exception as e:
                self.get_logger().error(f"Error processing image: {str(e)}")
                return None
        else:
            self.get_logger().warn("No RGB image available to process.")
            return None

def main(args=None):
    rclpy.init(args=args)
    node = MediaPipeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()