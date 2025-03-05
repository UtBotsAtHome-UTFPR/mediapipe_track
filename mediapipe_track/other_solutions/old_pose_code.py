#!/usr/bin/env python3

# ROS implementation of the Mediapipe Pose solution up to the 2022 libraries
# Inputs the sensor_msgs/Image message to feed the landmarks processing, outputs the skeletonImage, the detection status and the landmarks list

#!/usr/bin/env python3

import cv2
import mediapipe as mp
import numpy as np
from math import pow, sqrt, sin, tan, radians
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PointStamped, Point, TransformStamped
from sensor_msgs.msg import Image
from utbots_msgs.msg import Skeleton2d, Object
from utbots_actions.msg import MPPoseAction, MPPoseResult
from tf2_ros import TransformBroadcaster
from rclpy.action import ActionServer

class Camera():
    def __init__(self, fov_vertical, fov_horizontal, rgb_topic, depth_topic):
        self.fov_vertical = fov_vertical
        self.fov_horizontal = fov_horizontal
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic

class PersonPoseAction(Node):
    def __init__(self):
        super().__init__('person_pose')

        # Image FOV for trig calculations
        cameras = {
            "kinect": Camera(43, 57, "/camera/rgb/image_color", "/camera/depth_registered/image_raw"),
            "realsense": Camera(57, 86, "/camera/color/image_raw", "/camera/depth/image_raw")
        }

        # Parameters
        self.camera = self.declare_parameter('camera', 'realsense').value

        # Select camera
        try:
            selected_camera = cameras[self.camera]
        except:
            selected_camera = cameras["realsense"]

        self.camFov_vertical = selected_camera.fov_vertical
        self.camFov_horizontal = selected_camera.fov_horizontal
        self.camera_rgb_topic = selected_camera.rgb_topic
        self.camera_depth_topic = selected_camera.depth_topic

        # Tf
        self.tf_broadcaster = TransformBroadcaster(self)

        # Messages
        self.msg_rgbImg = None
        self.msg_depthImg = None
        self.msg_targetStatus = "Not Detected"
        self.msg_poseLandmarks = Skeleton2d()
        self.msg_targetSkeletonImg = Image()
        self.msg_selectedPerson = Object()
        self.msg_targetPoint = PointStamped()
        self.msg_targetPoint.header.frame_id = "target"

        # Publishers and Subscribers
        self.sub_rgbImg = self.create_subscription(
            Image,
            self.camera_rgb_topic,
            self.callback_rgbImg,
            10
        )
        self.sub_depthImg = self.create_subscription(
            Image,
            self.camera_depth_topic,
            self.callback_depthImg,
            10
        )

        self.pub_targetStatus = self.create_publisher(String, "pose/status", 10)
        self.pub_targetSkeletonImg = self.create_publisher(Image, "pose/skeletonImg", 10)
        self.pub_poseLandmarks = self.create_publisher(Skeleton2d, "pose/poseLandmarks", 10)
        self.pub_targetPoint = self.create_publisher(PointStamped, "selected/torsoPoint", 10)
        self.pub_tf = TransformBroadcaster(self)

        # Time
        self.loopRate = 30
        self.t_last = 0.0
        self.t_timeout = 2.250

        # Action server initialization
        self._as = ActionServer(self, MPPoseAction, 'mediapipe_pose', self.pose_action_callback)
        
        # Cv
        self.cvBridge = CvBridge()
        self.cv_img = None

        # Mediapipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose

        # Initialize pose model
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.75,
            min_tracking_confidence=0.9,
            model_complexity=2
        )

        # Main loop
        self.mainLoop()

# Callbacks
    def callback_rgbImg(self, msg):
        self.msg_rgbImg = msg

    def callback_depthImg(self, msg):
        self.msg_depthImg = msg
    
# Basic MediaPipe Pose methods
    def ProcessImg(self):
        # Check if the image message is available
        if self.msg_rgbImg is not None:
            try:
                # Convert the ROS image message to an OpenCV image
                cvImg = self.cvBridge.imgmsg_to_cv2(self.msg_rgbImg, "bgr8")

                # Set flags to prevent writing to the image, for performance
                cvImg.flags.writeable = False

                # Convert BGR to RGB (OpenCV default is BGR, but Mediapipe expects RGB)
                cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)

                return cvImg

            except Exception as e:
                self.get_logger().error(f"Error processing image: {str(e)}")
                return None
        else:
            self.get_logger().warn("No RGB image available to process.")
            return None
    
    def DrawLandmarks(self, cvImg, poseResults):
        # To draw the hand annotations on the image
        cvImg.flags.writeable = True

        # Back to BGR
        cvImg = cv2.cvtColor(cvImg, cv2.COLOR_RGB2BGR)

        # Draws the skeleton image
        self.mp_drawing.draw_landmarks(
            cvImg,
            poseResults.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        
        return self.cvBridge.cv2_to_imgmsg(cvImg)

# Body data processing
    def SetLandmarkPoints(self, landmark):
        landmarks = []
        for lmark in self.mp_pose.PoseLandmark:
            landmarks.append(self.PointByLmarks(landmark[lmark]))
        return landmarks

    def PointByLmarks(self, landmark):
        return Point(landmark.x, landmark.y, landmark.z)
        
    def CalculateTorsoCenter(self):
        # Gets the torso points coordinates
        right_shoulder = self.msg_poseLandmarks.points[self.msg_poseLandmarks.RIGHT_SHOULDER]
        left_shoulder  = self.msg_poseLandmarks.points[self.msg_poseLandmarks.LEFT_SHOULDER]
        right_hip      = self.msg_poseLandmarks.points[self.msg_poseLandmarks.RIGHT_HIP]
        left_hip       = self.msg_poseLandmarks.points[self.msg_poseLandmarks.LEFT_HIP]

        # Calculates the torso center point
        torsoCenter = Point(
                         (right_shoulder.x + left_shoulder.x + right_hip.x + left_hip.x)/4,
                         (right_shoulder.y + left_shoulder.y + right_hip.y + left_hip.y)/4,
                         (right_shoulder.z + left_shoulder.z + right_hip.z + left_hip.z)/4
                      )
        return torsoCenter

    def CropTorsoImg(self, img, img_encoding, torsoCenter):
        # Gets the torso points coordinates
        right_shoulder = self.msg_poseLandmarks.points[self.msg_poseLandmarks.RIGHT_SHOULDER]
        left_shoulder  = self.msg_poseLandmarks.points[self.msg_poseLandmarks.LEFT_SHOULDER]
        right_hip      = self.msg_poseLandmarks.points[self.msg_poseLandmarks.RIGHT_HIP]
 
        # Gets the image dimensions
        ## Depth image (32FC1) gives two dimensions
        if img_encoding == "32FC1":
            imageHeight, imageWidth = img.shape
        ## RGB image gives three dimensions (the third is color channel)
        else:
            imageHeight, imageWidth, _ = img.shape

        # Calculates the torso width and height
        torsoWidth = max(abs(right_shoulder.x - left_shoulder.x) * imageWidth, 1)
        torsoHeight = max(abs(right_shoulder.y - right_hip.y) * imageHeight, 1)

        # Calculates the torso frame coordinates
        x0 = max(int(torsoCenter.x * imageWidth - torsoWidth/2), 0)
        y0 = max(int(torsoCenter.y * imageHeight - torsoHeight/2), 0)
        xf = min(int(torsoCenter.x * imageWidth + torsoWidth/2), imageWidth)
        yf = min(int(torsoCenter.y * imageHeight + torsoHeight/2), imageHeight)

        # Crops the torso
        cropped_image = img[y0:yf, x0:xf]
        return cropped_image

    # Calculates mean distance to the camera of all torso pixels
    def GetTorsoDistance(self, croppedDepthImg):
        height, width = croppedDepthImg.shape
        npArray = croppedDepthImg[0:height, 0:width]

        rowMeans = np.array([])
        for row in npArray:
            rowMeans = np.append(rowMeans, np.mean(row))

        depthMean = np.mean(rowMeans)
        return depthMean

# Calculates the 3D point of a depth pixel by using rule of three and considering the FOV of the camera:
    def Get3dPointFromDepthPixel(self, pixel, distance):
        # The height and width are already scaled to 0-1 both by Mediapipe
        width  = 1.0
        height = 1.0

        # Centralize the camera reference at (0,0,0)
        ## (x,y,z) are respectively horizontal, vertical and depth
        ## Theta is the angle of the point with z axis in the zx plane
        ## Phi is the angle of the point with z axis in the zy plane
        ## x_max is the distance of the side border from the camera
        ## y_max is the distance of the upper border from the camera
        theta_max = self.camFov_horizontal/2 
        phi_max = self.camFov_vertical/2
        x_max = width/2.0
        y_max = height/2.0
        x = pixel.x - x_max
        y = pixel.y - y_max

        # Caculate point theta and phi
        theta = radians(theta_max * x / x_max)
        phi = radians(phi_max * y / y_max)

        # Convert the spherical radius rho from Kinect's mm to meter
        rho = distance/1000

        # Calculate x, y and z
        y = rho * sin(phi)
        x = sqrt(pow(rho, 2) - pow(y, 2)) * sin(theta)
        z = x / tan(theta)

        # Change coordinate scheme
        ## We calculate with (x,y,z) respectively horizontal, vertical and depth
        ## For the plot in 3d space, we need to remap the coordinates to (z, -x, -y)
        point_zxy = Point(z, -x, -y)

        return point_zxy
    
# Transformation tree methods
    def SetupTfMsg(self, x, y, z):
        msg_tfStamped = TransformStamped()
        msg_tfStamped.header.frame_id = "camera_link"
        msg_tfStamped.header.stamp = self.get_clock().now().to_msg()  # Get current time in ROS 2 format
        msg_tfStamped.child_frame_id = "target"
        msg_tfStamped.transform.translation.x = 0
        msg_tfStamped.transform.translation.y = 0
        msg_tfStamped.transform.translation.z = 0
        msg_tfStamped.transform.rotation.x = 0.0
        msg_tfStamped.transform.rotation.y = 0.0
        msg_tfStamped.transform.rotation.z = 0.0
        msg_tfStamped.transform.rotation.w = 1.0

        # Broadcast transformations
        self.tf_broadcaster.sendTransform(msg_tfStamped)

# Nodes Publish
    def PublishEverything(self):
        self.pub_targetSkeletonImg.publish(self.msg_targetSkeletonImg)
        self.pub_poseLandmarks.publish(self.msg_poseLandmarks)
        self.pub_targetStatus.publish(self.msg_targetStatus)
        self.pub_targetPoint.publish(self.msg_targetPoint)
        self.SetupTfMsg(self.msg_targetPoint.point.x, self.msg_targetPoint.point.y, self.msg_targetPoint.point.z)

    def pose_action(self, goal):
        self.get_logger().info("[MPPOSE] Goal recieved")
        # Optional specific image sent as goal
        # if goal.Image.width != 0 and goal.Image.height != 0:
        #     print( goal.Image.width )
        #     cvImg = self.bridge.imgmsg_to_cv2(goal.Image, desired_encoding="bgr8")
        # else:
        
        # try:
        # Topic subscribed image
        cvImg = self.ProcessImg()
            
        # Process the results
        poseResults = self.pose.process(cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB))

        self.msg_targetSkeletonImg = self.DrawLandmarks(cvImg, poseResults)

        # Manage action results
        action_res = MPPoseResult()
        action_res.Success = Bool()
    
        # Processes pose results 
        if poseResults.pose_landmarks:
            # Pose Landmarks are normalized image coordinates
            self.msg_poseLandmarks.points = self.SetLandmarkPoints(poseResults.pose_landmarks.landmark)
            self.msg_targetStatus = "Detected"

            # Calculate torso center point
            torsoCenter = self.CalculateTorsoCenter()

            # Converts the rgb image to OpenCV format, so it can be manipulated (32FC1 depth encoding)
            depth_img_encoding = "32FC1"
            cv_depthImg = self.cvBridge.imgmsg_to_cv2(self.msg_depthImg, depth_img_encoding)

            # Try to crop the depth torso image and calculate 3d torso point
            try:
                croppedDepthImg = self.CropTorsoImg(cv_depthImg, depth_img_encoding, torsoCenter)
                self.msg_targetPoint.point = self.Get3dPointFromDepthPixel(torsoCenter, self.GetTorsoDistance(croppedDepthImg))

                action_res.Point = self.msg_targetPoint
                action_res.Success.data = True

                # Point creation timestamp
                t_now = self.get_clock().now().seconds()
                self.t_last = t_now
            except:
                self.get_logger().error("------------- Error in depth crop -------------")
                # return 0               
        else:
            self.msg_targetStatus = "Not Detected"
            t_now = self.get_clock().now().seconds()
            # Evaluates time interval from the last detected person point calculated
            # Sets the point values to origin if the interval is bigger than the defined timeout
            if (t_now - self.t_last > self.t_timeout):
                self.t_last = t_now
                self.msg_targetPoint.point = Point(0, 0, 0)
            action_res.Success.data = False
        self._as.set_succeeded(action_res)

        # Console logs
        self.get_logger().info("[MPPOSE] status: {}".format(self.msg_targetStatus))
        self.get_logger().info("[MPPOSE] xyz: ({}, {}, {})".format(
            self.msg_targetPoint.point.x, 
            self.msg_targetPoint.point.y, 
            self.msg_targetPoint.point.z))

        self.PublishEverything()

        # except:
        #     self.get_logger().info("Not ready yet")
        #     self._as.set_aborted(action_res)

# Main
def main(args=None):
    rclpy.init(args=args)
    person_pose = PersonPoseAction()
    rclpy.spin(person_pose)
    rclpy.shutdown()

if __name__ == '__main__':
    main()