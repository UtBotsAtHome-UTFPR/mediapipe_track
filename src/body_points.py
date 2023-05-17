#!/usr/bin/env python3

# Image processing
from cv_bridge import CvBridge

# ROS
import rospy
## Message definitions
from std_msgs.msg import String
from std_msgs.msg import Bool
from geometry_msgs.msg import PointStamped, Point, TransformStamped
#from hri_msgs.msg import Skeleton2D
from vision_msgs.msg import PointArray
from sensor_msgs.msg import Image
## Transformation tree
from tf.msg import tfMessage
from geometry_msgs.msg import TransformStamped
# Math 
import numpy as np
from math import pow, sqrt, sin, cos, tan, radians

class BodyPoints():
    def __init__(self, topic_rgbImg, topic_depthImg, camFov_vertical, camFov_horizontal):
        # Image FOV for trig calculations
        self.camFov_vertical = camFov_vertical
        self.camFov_horizontal = camFov_horizontal

        # Messages
        self.msg_tfStamped                   = TransformStamped()
        self.msg_targetPoint                 = PointStamped()   # Point
        self.msg_targetPoint.header.frame_id = "target"
        self.msg_targetCroppedRgbTorso       = Image()
        self.msg_targetCroppedDepthTorso     = Image()
        self.msg_rgbImg                      = None      # Image
        self.msg_depthImg                    = None      # Image
        self.msg_poseLandmarks               = PointArray()

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
            "/utbots/vision/person/pose/poseLandmarks", PointArray, self.callback_poseLandmarks)
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
        # self.pub_targetStatus = rospy.Publisher(
        #     "/utbots/vision/lock/status", String, queue_size=10)

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
        # print("- RGB: new msg")

    def callback_depthImg(self, msg):
        self.msg_depthImg = msg
        self.newDepthImg = True
        # print("- Depth: new msg")
    
    def callback_poseLandmarks(self, msg):
        self.msg_poseLandmarks = msg
        self.newPoseLandmarks = True

    def callback_targetStatus(self, msg):
        self.msg_targetStatus = msg

## Torso
    def GetTorsoPoints(self, landmarks):
        return [landmarks[12], landmarks[11], landmarks[24], landmarks[23]]

    def CropTorsoImg(self, img, imgEncoding, torsoPoints, torsoCenter):
        if imgEncoding == "32FC1":
            imageHeight, imageWidth = img.shape
        else:
            imageHeight, imageWidth, a = img.shape
        torsoWidth = max(abs(torsoPoints[0].x - torsoPoints[1].x) * imageWidth, 1)
        torsoHeight = max(abs(torsoPoints[0].y - torsoPoints[2].y) * imageHeight, 1)

        x0 = max(int(torsoCenter.x * imageWidth - torsoWidth/2), 0)
        y0 = max(int(torsoCenter.y * imageHeight - torsoHeight/2), 0)
        xf = min(int(torsoCenter.x * imageWidth + torsoWidth/2), imageWidth)
        yf = min(int(torsoCenter.y * imageHeight + torsoHeight/2), imageHeight)

        cropped_image = img[y0:yf, x0:xf]
        return cropped_image

    def GetTorsoDistance(self, croppedDepthImg):
        height, width = croppedDepthImg.shape
        npArray = croppedDepthImg[0:height, 0:width]

        rowMeans = np.array([])
        for row in npArray:
            rowMeans = np.append(rowMeans, np.mean(row))

        depthMean = np.mean(rowMeans)
        return depthMean

    def ProcessTorso(self):
        
        if self.newPoseLandmarks == True and self.msg_targetStatus.data == "Detected":
            self.newPoseLandmarks = False

            torsoPoints = self.GetTorsoPoints(self.msg_poseLandmarks.points)
            torsoCenter = self.GetPointsMean(torsoPoints)

            if self.newRgbImg == True:
                self.newRgbImg = False
                cv_rbgImg = self.cvBridge.imgmsg_to_cv2(self.msg_rgbImg, "bgr8")
                try:
                    croppedRgbImg = self.CropTorsoImg(cv_rbgImg, "passthrough", torsoPoints, torsoCenter)
                    self.msg_targetCroppedRgbTorso = self.cvBridge.cv2_to_imgmsg(croppedRgbImg)
                except:
                    rospy.loginfo("------------- Error in RGB crop -------------")
                    return 0

            if self.newDepthImg == True:
                self.newDepthImg = False
                cv_depthImg = self.cvBridge.imgmsg_to_cv2(self.msg_depthImg, "32FC1")
                try:
                    croppedDepthImg = self.CropTorsoImg(cv_depthImg, "32FC1", torsoPoints, torsoCenter)
                    self.msg_targetCroppedDepthTorso = self.cvBridge.cv2_to_imgmsg(croppedDepthImg)
                    torsoCenter3d = self.ExtractDepthPoint(torsoCenter, self.GetTorsoDistance(croppedDepthImg))
                    self.msg_targetPoint.point = torsoCenter3d

                    t_now = rospy.get_time()
                    self.t_last = t_now
                except:
                    rospy.loginfo("------------- Error in depth crop -------------")
                    return 0                
        # Nothing detected...
        else:
            t_now = rospy.get_time()
            if (t_now - self.t_last > self.t_timeout):
                self.t_last = t_now
                self.msg_targetPoint.point = Point(0, 0, 0)

        rospy.loginfo("\nTARGET")
        rospy.loginfo(" - status: {}".format(self.msg_targetStatus))
        rospy.loginfo(" - xyz: ({}, {}, {})".format(
            self.msg_targetPoint.point.x, 
            self.msg_targetPoint.point.y, 
            self.msg_targetPoint.point.z))

# Points calculations
    def GetPointsMean(self, points):
        sum_x = 0
        sum_y = 0
        sum_z = 0
        counter = 0
        for point in points:
            sum_x = sum_x + point.x
            sum_y = sum_y + point.y
            sum_z = sum_z + point.z
            counter = counter + 1
        return Point(sum_x/counter, sum_y/counter, sum_z/counter)

    ''' By using rule of three and considering the FOV of the camera:
            - Calculates the 3D point of a depth pixel '''
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

    ''' Transforms the mpipe coordinate format to tf tree coordinate format'''
    def XyzToZxy(self, point):
        return Point(point.z, point.x, point.y)   

    def ExtractDepthPoint(self, coordinate, mean_dist):
        depthPoint = self.Get3dPointFromDepthPixel(coordinate, mean_dist)
        return self.XyzToZxy(depthPoint)

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
            self.PublishEverything()
            self.ProcessTorso()
              
if __name__ == "__main__":
    BodyPoints(
        "/camera/rgb/image_raw",
        "/camera/depth/image_raw",
        43,
        57)
