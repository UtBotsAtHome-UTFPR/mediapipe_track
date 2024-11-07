#!/usr/bin/env python3

# ROS implementation of the Mediapipe Pose solution up to the 2022 libraries
# Inputs the sensor_msgs/Image message to feed the landmarks processing, outputs the skeletonImage, the detection status and the landmarks list

# Mediapipe imports
import cv2
import mediapipe as mp

# Math 
import numpy as np
from math import pow, sqrt, sin, tan, radians

# Image processing
from cv_bridge import CvBridge

# ROS
import rospy
import actionlib
## Message definitions
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PointStamped, Point, TransformStamped
from sensor_msgs.msg import Image
from vision_msgs.msg import Skeleton2d, Object
from utbots_actions.msg import MPPoseAction, MPPoseResult
## Transformation tree
from tf.msg import tfMessage

class Camera():
    def __init__(self, fov_vertical, fov_horizontal, rgb_topic, depth_topic):
        self.fov_vertical = fov_vertical
        self.fov_horizontal = fov_horizontal
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic

class PersonPoseAction():
    def __init__(self):
        # Image FOV for trig calculations
        cameras = {
            "kinect": Camera(43, 57, "/camera/rgb/image_color", "/camera/depth_registered/image_raw"),
            "realsense": Camera(57, 86, "/camera/color/image_raw", "/camera/depth/image_raw")
        }

        # ROS node
        rospy.init_node('person_pose', anonymous=True)

        # Parameters
        self.camera = rospy.get_param('~camera', 'realsense')

        # Select camera
        try:
            selected_camera = cameras[self.camera]
        except:
            selected_camera = cameras["realsense"]

        self.camFov_vertical = selected_camera.fov_vertical
        self.camFov_horizontal = selected_camera.fov_horizontal
        self.camera_rgb_topic = selected_camera.rgb_topic
        self.camera_depth_topic = selected_camera.depth_topic

        # Messages
        self.msg_rgbImg                      = None             # Image
        self.msg_depthImg                    = None             # Image
        self.msg_tfStamped                   = TransformStamped()
        self.msg_targetStatus                = "Not Detected"
        self.msg_targetCroppedDepthTorso     = Image()
        self.msg_poseLandmarks               = Skeleton2d()
        self.msg_targetSkeletonImg           = Image()
        self.msg_selectedPerson              = Object()
        self.msg_targetPoint                 = PointStamped()   # Point
        self.msg_targetPoint.header.frame_id = "target"

        # Publishers and Subscribers
        self.sub_rgbImg = rospy.Subscriber(
            self.camera_rgb_topic, Image, self.callback_rgbImg)
        self.sub_depthImg = rospy.Subscriber(
            self.camera_depth_topic, Image, self.callback_depthImg)

        self.pub_targetStatus = rospy.Publisher(
            "pose/status", String, queue_size=1)  
        self.pub_targetSkeletonImg = rospy.Publisher(
            "pose/skeletonImg", Image, queue_size=1)
        self.pub_poseLandmarks = rospy.Publisher(
            "pose/poseLandmarks", Skeleton2d, queue_size=1)
        self.pub_targetCroppedDepthTorso = rospy.Publisher(
            "selected/croppedTorso/depth", Image, queue_size=1)
        self.pub_targetPoint = rospy.Publisher(
            "selected/torsoPoint", PointStamped, queue_size=1)
        self.pub_tf = rospy.Publisher(
            "/tf", tfMessage, queue_size=1)


        # Time
        self.loopRate = rospy.Rate(30)
        self.t_last = 0.0  # sec
        self.t_timeout = 2.250  # sec

        # Action server initialization
        self._as = actionlib.SimpleActionServer('mediapipe_pose', MPPoseAction, self.pose_action, False)
        self._as.start()

        # Cv
        self.cvBridge = CvBridge()
        self.cv_img = None

        # Mediapipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose

        # Calls main loop
        self.pose = self.mp_pose.Pose(
            # Pose Configurations
            min_detection_confidence=0.75,
            min_tracking_confidence=0.9,
            model_complexity=2
            )
        self.mainLoop()

# Callbacks
    def callback_rgbImg(self, msg):
        self.msg_rgbImg = msg

    def callback_depthImg(self, msg):
        self.msg_depthImg = msg
    
# Basic MediaPipe Pose methods
    def ProcessImg(self):
        # Conversion to cv image
        cvImg = self.cvBridge.imgmsg_to_cv2(self.msg_rgbImg, "bgr8")

        # Not writeable passes by reference (better performance)
        cvImg.flags.writeable = False

        # Converts BGR to RGB
        cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)

        return cvImg
    
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
        self.msg_tfStamped.header.frame_id = "camera_link"
        self.msg_tfStamped.header.stamp = rospy.Time.now()
        self.msg_tfStamped.child_frame_id = "target"
        self.msg_tfStamped.transform.translation.x = 0
        self.msg_tfStamped.transform.translation.y = 0
        self.msg_tfStamped.transform.translation.z = 0
        self.msg_tfStamped.transform.rotation.x = 0.0
        self.msg_tfStamped.transform.rotation.y = 0.0
        self.msg_tfStamped.transform.rotation.z = 0.0
        self.msg_tfStamped.transform.rotation.w = 1.0

        msg_tf = tfMessage([self.msg_tfStamped])
        self.pub_tf.publish(msg_tf)

# Nodes Publish
    def PublishEverything(self):
        self.pub_targetSkeletonImg.publish(self.msg_targetSkeletonImg)
        self.pub_poseLandmarks.publish(self.msg_poseLandmarks)
        self.pub_targetStatus.publish(self.msg_targetStatus)
        self.pub_targetCroppedDepthTorso.publish(self.msg_targetCroppedDepthTorso)
        self.pub_targetPoint.publish(self.msg_targetPoint)
        self.SetupTfMsg(self.msg_targetPoint.point.x, self.msg_targetPoint.point.y, self.msg_targetPoint.point.z)

    def pose_action(self, goal):
        rospy.loginfo("[MPPOSE] Goal recieved")
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
                self.msg_targetCroppedDepthTorso = self.cvBridge.cv2_to_imgmsg(croppedDepthImg)
                self.msg_targetPoint.point = self.Get3dPointFromDepthPixel(torsoCenter, self.GetTorsoDistance(croppedDepthImg))

                action_res.Point = self.msg_targetPoint
                action_res.Success.data = True

                # Point creation timestamp
                t_now = rospy.get_time()
                self.t_last = t_now
            except:
                rospy.logerr("------------- Error in depth crop -------------")
                # return 0               
        else:
            self.msg_targetStatus = "Not Detected"
            t_now = rospy.get_time()
            # Evaluates time interval from the last detected person point calculated
            # Sets the point values to origin if the interval is bigger than the defined timeout
            if (t_now - self.t_last > self.t_timeout):
                self.t_last = t_now
                self.msg_targetPoint.point = Point(0, 0, 0)
            action_res.Success.data = False
        self._as.set_succeeded(action_res)

        # Console logs
        rospy.loginfo("[MPPOSE] status: {}".format(self.msg_targetStatus))
        rospy.loginfo("[MPPOSE] xyz: ({}, {}, {})".format(
            self.msg_targetPoint.point.x, 
            self.msg_targetPoint.point.y, 
            self.msg_targetPoint.point.z))

        self.PublishEverything()

        # except:
        #     rospy.loginfo("Not ready yet")
        #     self._as.set_aborted(action_res)

# Main
    def mainLoop(self):
        while rospy.is_shutdown() == False:
            self.loopRate.sleep()
    
if __name__ == "__main__":
    PersonPoseAction()
