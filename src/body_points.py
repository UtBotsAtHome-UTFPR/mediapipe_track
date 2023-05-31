#!/usr/bin/env python3

# Through MP Pose landmarks and RGB-D images estimates a 3d point (geometry_msgs/PointStamped) for the detected person from the camera reference 

# Image processing
from cv_bridge import CvBridge

# ROS
import rospy
## Message definitions
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, Point, TransformStamped
from vision_msgs.msg import Skeleton2d
from sensor_msgs.msg import Image
## Transformation tree
from tf.msg import tfMessage
from geometry_msgs.msg import TransformStamped

# Math 
import numpy as np
from math import pow, sqrt, sin, tan, radians

class BodyPoints():
    def __init__(self, topic_rgbImg, topic_depthImg, camFov_vertical, camFov_horizontal):
        # Image FOV for trig calculations
        self.camFov_vertical = camFov_vertical
        self.camFov_horizontal = camFov_horizontal

        # Messages
        self.msg_tfStamped                   = TransformStamped()
        self.msg_targetStatus                = "Not Detected"
        self.msg_targetPoint                 = PointStamped()   # Point
        self.msg_targetPoint.header.frame_id = "target"
        self.msg_targetCroppedRgbTorso       = Image()
        self.msg_targetCroppedDepthTorso     = Image()
        self.msg_rgbImg                      = None             # Image
        self.msg_depthImg                    = None             # Image
        self.msg_poseLandmarks               = Skeleton2d()

        # To tell if there's a new msg
        self.newRgbImg = False
        self.newDepthImg = False
        self.newPoseLandmarks = False

        # Publishers and Subscribers
        self.sub_rgbImg = rospy.Subscriber(
            topic_rgbImg, Image, self.callback_rgbImg)
        self.sub_depthImg = rospy.Subscriber(
            topic_depthImg, Image, self.callback_depthImg)
        self.sub_poseLandmarks = rospy.Subscriber(
            "/utbots/vision/person/pose/poseLandmarks", Skeleton2d, self.callback_poseLandmarks)
        self.sub_targetStatus = rospy.Subscriber(
            "/utbots/vision/person/pose/status", String, self.callback_targetStatus)

        self.pub_tf = rospy.Publisher(
            "/tf", tfMessage, queue_size=1)
        self.pub_targetPoint = rospy.Publisher(
            "/utbots/vision/person/selected/torsoPoint", PointStamped, queue_size=10)
        self.pub_targetCroppedRgbTorso = rospy.Publisher(
            "/utbots/vision/person/selected/croppedTorso/rgb", Image, queue_size=10)
        self.pub_targetCroppedDepthTorso = rospy.Publisher(
            "/utbots/vision/person/selected/croppedTorso/depth", Image, queue_size=10)

        # ROS node
        rospy.init_node('body_points', anonymous=True)

        # Time
        self.loopRate = rospy.Rate(30)
        self.t_last = 0.0  # sec
        self.t_timeout = 2.250  # sec

        # Cv
        self.cvBridge = CvBridge()
        self.mainLoop()

# Callbacks
    def callback_rgbImg(self, msg):
        self.msg_rgbImg = msg
        self.newRgbImg = True

    def callback_depthImg(self, msg):
        self.msg_depthImg = msg
        self.newDepthImg = True
    
    def callback_poseLandmarks(self, msg):
        self.msg_poseLandmarks = msg
        self.newPoseLandmarks = True

    def callback_targetStatus(self, msg):
        self.msg_targetStatus = msg

## Torso methods

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
        self.pub_targetCroppedRgbTorso.publish(self.msg_targetCroppedRgbTorso)
        self.pub_targetCroppedDepthTorso.publish(self.msg_targetCroppedDepthTorso)
        self.pub_targetPoint.publish(self.msg_targetPoint)
        self.SetupTfMsg(self.msg_targetPoint.point.x, self.msg_targetPoint.point.y, self.msg_targetPoint.point.z)

# Main
    def mainLoop(self):
        while rospy.is_shutdown() == False:
            self.loopRate.sleep()
        
            if self.newPoseLandmarks == True and self.msg_targetStatus.data == "Detected":
                self.newPoseLandmarks = False

                # Calculate torso center point
                torsoCenter = self.CalculateTorsoCenter()

                if self.newRgbImg == True:
                    self.newRgbImg = False

                    rgb_img_encoding = "bgr8"
                    # Converts the rgb image to OpenCV format, so it can be manipulated (bgr8 encoding)
                    cv_rbgImg = self.cvBridge.imgmsg_to_cv2(self.msg_rgbImg, rgb_img_encoding)

                    # Try to crop the rgb torso image
                    try:
                        croppedRgbImg = self.CropTorsoImg(cv_rbgImg, rgb_img_encoding, torsoCenter)
                        self.msg_targetCroppedRgbTorso = self.cvBridge.cv2_to_imgmsg(croppedRgbImg)
                    except:
                        rospy.loginfo("------------- Error in RGB crop -------------")
                        # return 0

                if self.newDepthImg == True:
                    self.newDepthImg = False

                    # Converts the rgb image to OpenCV format, so it can be manipulated (32FC1 depth encoding)
                    depth_img_encoding = "32FC1"
                    cv_depthImg = self.cvBridge.imgmsg_to_cv2(self.msg_depthImg, depth_img_encoding)

                    # Try to crop the depth torso image and calculate 3d torso point
                    try:
                        croppedDepthImg = self.CropTorsoImg(cv_depthImg, depth_img_encoding, torsoCenter)
                        self.msg_targetCroppedDepthTorso = self.cvBridge.cv2_to_imgmsg(croppedDepthImg)
                        self.msg_targetPoint.point = self.Get3dPointFromDepthPixel(torsoCenter, self.GetTorsoDistance(croppedDepthImg))

                        # Point creation timestamp
                        t_now = rospy.get_time()
                        self.t_last = t_now
                    except:
                        rospy.loginfo("------------- Error in depth crop -------------")
                        # return 0               

            # Nothing detected
            else:
                t_now = rospy.get_time()
                # Evaluates time interval from the last detected person point calculated
                # Sets the point values to origin if the interval is bigger than the defined timeout
                if (t_now - self.t_last > self.t_timeout):
                    self.t_last = t_now
                    self.msg_targetPoint.point = Point(0, 0, 0)

            # Console logs
            rospy.loginfo("\nTARGET")
            rospy.loginfo(" - status: {}".format(self.msg_targetStatus))
            rospy.loginfo(" - xyz: ({}, {}, {})".format(
                self.msg_targetPoint.point.x, 
                self.msg_targetPoint.point.y, 
                self.msg_targetPoint.point.z))

            self.PublishEverything()
              
if __name__ == "__main__":
    BodyPoints(
        "/camera/rgb/image_raw",
        "/camera/depth_registered/image_raw",
        43,
        57)
