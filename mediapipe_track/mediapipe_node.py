#!/usr/bin/python3

import mediapipe as mp
from .mediapipe_pose import MediaPipePose
import numpy as np
import cv2
import os
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from utbots_actions.action import MPPose
from sensor_msgs.msg import Image
from utbots_msgs.msg import Skeleton2d
from geometry_msgs.msg import PointStamped, Point, TransformStamped
from std_msgs.msg import String, Bool
from std_srvs.srv import SetBool
from tf2_ros import TransformBroadcaster
import ament_index_python.packages

package_path = ament_index_python.packages.get_package_share_directory("mediapipe_track")
package_root = os.path.dirname(package_path)

class MediaPipeNode(Node, MediaPipePose):
    """
    A ROS2 node that processes RGB image messages using MediaPipe Pose estimation
    and publishes detected skeleton landmarks, torso points, and overlay images.

    ## Parameters:
    - `topic_namespace` (string)
    Namespace for published topics.
    - `model_path` (string)
    Path of the selected MediaPipe model. => Inherited from MediaPipePose
    - `num_poses` (int)
    Maximum number of poses to detect. => Inherited from MediaPipePose
    - `detection_conf` (float)
    Confidence threshold for detecting a pose. => Inherited from MediaPipePose
    - `presence_conf` (float)
    Confidence threshold for presence estimation. => Inherited from MediaPipePose
    - `track_conf` (float)
    Confidence threshold for pose tracking. => Inherited from MediaPipePose
    - `segmentation_mask` (bool)
    Whether to include segmentation mask output. => Inherited from MediaPipePose
    - `draw_skeleton_img` (string)
    Enable drawing skeleton image for synchronous processing.
    - `calculate_torso_point` (string)
    Enable torso point estimation for synchronous processing.
    - `rgb_topic` (string,)
    Input topic for RGB images.

    ## Publishers:
    - `pose/skeleton_img` (sensor_msgs/Image)
    Publishes a visualized image of detected poses.
    - `pose/pose_landmarks` (utbots_msgs/Skeleton2d)
    Publishes detected 2D skeleton landmarks.
    - `pose/torso_point` (geometry_msgs/PointStamped)
    Publishes the detected torso point.
    - `tf` (tf2_msgs/TFMessage)
    Broadcasts torso transform over TF.

    ## Subscribers:
    - `<rgb_topic>` (sensor_msgs/Image)
    Subscribes to an RGB image stream for processing.

    ## Services:
    - `enable_synchronous` (std_srvs/SetBool)
    Enables or disables synchronous pose processing.

    ## Actions:
    - `mediapipe_pose` (utbots_actions/MPPose)
    Action server for processing an image and returning pose landmarks.

    """
    def __init__(self):
        Node.__init__(self, 'mediapipe_node')
        MediaPipePose.__init__(self)

        # Set parameters
        self.declare_parameter('topic_namespace', "/utbots/vision")
        self.declare_parameter('model_path', '')
        self.declare_parameter('num_poses', 1)
        self.declare_parameter('detection_conf', 0.75)
        self.declare_parameter('presence_conf', 0.5)
        self.declare_parameter('track_conf', 0.9)
        self.declare_parameter('segmentation_mask', False)
        self.declare_parameter('draw_skeleton_img', True)
        self.declare_parameter('calculate_torso_point', True)
        self.declare_parameter('rbg_topic', "/image_raw")

        self.topic_namespace = self.get_parameter('topic_namespace').get_parameter_value().string_value
        self.num_poses = self.get_parameter('num_poses').get_parameter_value().integer_value
        self.detection_conf = self.get_parameter('detection_conf').get_parameter_value().double_value
        self.presence_conf = self.get_parameter('presence_conf').get_parameter_value().double_value
        self.track_conf = self.get_parameter('track_conf').get_parameter_value().double_value
        self.segmentation_mask = self.get_parameter('segmentation_mask').get_parameter_value().bool_value
        self.draw_param = self.get_parameter('draw_skeleton_img').get_parameter_value().bool_value
        self.torso_param = self.get_parameter('calculate_torso_point').get_parameter_value().bool_value
        self.rgb_topic = self.get_parameter('rbg_topic').get_parameter_value().string_value
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        if model_path == '':
            self.model_path=package_path + "/models/pose_landmarker_lite.task"
        else:
            self.model_path=model_path

        # Define ROS messages
        self.msg_pose_landmarks = Skeleton2d()
        self.msg_target_skeleton_img = Image()
        self.msg_target_point = PointStamped()
        self.msg_rgb_img = Image()

        # Publishers and Subscribers
        self.sub_rgbImg = self.create_subscription(
            Image,
            self.rgb_topic,
            self.callback_rgb_img,
            10
        )
        self.pub_target_skeletonImg = self.create_publisher(Image, self.topic_namespace + "pose/skeleton_img", 10)
        self.pub_poseLandmarks = self.create_publisher(Skeleton2d, self.topic_namespace + "pose/pose_landmarks", 10)
        self.pub_target_point = self.create_publisher(PointStamped, self.topic_namespace + "pose/torso_point", 10)
        self.pub_tf = TransformBroadcaster(self)

        # OpenCV image format conversion
        self.cvBridge = CvBridge()
        self.cv_img = None

        # Service to enable/disable synchronous processing
        self.srv_enable_sync = self.create_service(
            SetBool,
            'enable_synchronous',
            self.enable_synchronous_callback
        )
        self.enable_synchronous = False

        # Action server initialization
        self._as = ActionServer(self, MPPose, 'mediapipe_pose', self.pose_action_callback)
        
        # Timer for synchronous processing
        self.timer = self.create_timer(0.1, self.synchronous_callback)

    # Callbacks
    def callback_rgb_img(self, msg):
        """ RBG image topic callback """
        self.msg_rgb_img = msg

    def enable_synchronous_callback(self, request, response):
        """ Enable sychronous service callback """
        self.enable_synchronous = request.data
        response.success = True
        response.message = f"Synchronous processing {'enabled' if self.enable_synchronous else 'disabled'}."
        return response

    def pose_action_callback(self, goal):
        """ Single estimation asynchronous action callback """
        self.get_logger().info("[MPPOSE] Goal received")

        # Optional specific image sent as goal
        if goal.Image.width != 0 and goal.Image.height != 0:
            img_msg = goal.Image
        else:
            img_msg = self.msg_rgb_img

        # Manage action results
        action_res = MPPose.Result()
        action_res.success = Bool()

        pose_landmarks, skeleton_img, target_point = self.preprocess_and_predict(img_msg, draw=goal.GetDrawn, torso_point=goal.GetTorsoPoint)

        # Publish results to topic and to action
        if pose_landmarks:
            self.pub_poseLandmarks.publish(pose_landmarks)
            
            # Only considers TorsoPoint and Drawn if requested by Goal
            if goal.GetTorsoPoint:
                action_res.point = target_point
                self.pub_target_point.publish(target_point)

            if goal.GetDrawn:
                self.pub_target_skeletonImg.publish(skeleton_img)
                action_res.skeleton_img = skeleton_img
            
            action_res.success = True
            self._as.set_succeeded(action_res)
        else:
            action_res.success = False
            self._as.set_aborted()

    # Methods
    def synchronous_callback(self):
        """ Synchronous processing callback, enabled/disabled by a service """
        if self.enable_synchronous and self.msg_rgb_img is not None:
            pose_landmarks, skeleton_img, torso_point = self.preprocess_and_predict(self.msg_rgb_img, draw=self.draw_param, torso_point=False)
            self.pub_poseLandmarks.publish(pose_landmarks)
            if self.draw_param:
                self.pub_target_skeletonImg.publish(skeleton_img)
            if self.torso_param:
                self.pub_target_point.publish(torso_point)
            

    def preprocess_and_predict(self, img_msg, draw=False, torso_point=False):
        """ Converts the img_msg to cv format, makes a landmark estimation with MediaPipe Pose and returns the results"""
        if img_msg.height > 0 and img_msg.width > 0:
            cv_img = self.imgmsg_to_cv(img_msg)
            pose_landmarks_list, drawn_img = self.predict_landmarks(cv_img, draw=draw)

            points_array = []
            if pose_landmarks_list:
                pose_landmarks = pose_landmarks_list[0]
                for landmark in pose_landmarks:
                    point = Point()
                    point.x = landmark.x
                    point.y = landmark.y
                    point.z = landmark.z
                    points_array.append(point)

            self.msg_pose_landmarks.points = points_array

            # Draw landmarks in a skeleton img if requested
            if draw:
                self.msg_target_skeleton_img = self.cvBridge.cv2_to_imgmsg(drawn_img)
            
            # Calculate the torso point if requested
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
        """ If the image message is available, tries to convert to RGB8 for processing"""
        if img_msg is not None:
            try:
                # Convert the ROS image message to an OpenCV image
                cv_img = self.cvBridge.imgmsg_to_cv2(img_msg, "bgr8")
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
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