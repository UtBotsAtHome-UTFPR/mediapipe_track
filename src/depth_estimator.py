#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from darknet_ros_msgs.msg import BoundingBoxes
from math import radians, sqrt, tan
import numpy as np
import cv2
from cv_bridge import CvBridge

class DepthEstimatorNode():
    boundingBox = None

    def __init__(self, topic_point, topic_depthImg, topic_boundingBox, camFov_vertical, camFov_horizontal):
        # Image FOV for trig calculations
        self.camFov_vertical = camFov_vertical
        self.camFov_horizontal = camFov_horizontal
        
        # Published 3D point
        self.msg_point = None
        self.pub_point = rospy.Publisher(
            topic_point, 
            Point, 
            queue_size=0)

        # Subscribed depth img
        self.msg_depthImg = None
        self.newDepthImg = False
        self.sub_depthImg = rospy.Subscriber(
            topic_depthImg,
            Image,
            self.callback_depthImg,
            queue_size=1)

        # Subscribed bounding box
        self.boundingBox.msg = None
        self.boundingBox.center = None
        self.boundingBox.x_size = 0
        self.boundingBox.y_size = 0
        self.newBoundingBox = False
        self.sub_boundingBox = rospy.Subscriber(
            topic_boundingBox,
            BoundingBoxes,
            self.callback_boundingBox,
            queue_size=1)

        # Cv
        self.cvBridge = CvBridge()

        # ROS node
        rospy.init_node('depth_estimator', anonymous=True)

        # Time
        self.loopRate = rospy.Rate(30)

        # Loop
        rospy.loginfo("Starting loop...")
        self.mainLoop()

    def callback_depthImg(self, msg):
        self.msg_depthImg = msg
        self.newDepthImg = True
        # rospy.loginfo("- Depth: new msg")

    def callback_boundingBox(self, msg):
        self.boundingBox.msg = msg
        self.newBoundingBox = True
        rospy.loginfo("- RGB: new bounding box array")
        for box in msg.bounding_boxes:
            rospy.loginfo(
                "Xmin: {}, Xmax: {} Ymin: {}, Ymax: {}, Center: {}:{}".format(
                    box.xmin, box.xmax, box.ymin, box.ymax, int((box.xmax + box.xmin)/2), int((box.ymax + box.ymin)/2)
                )
            )
            self.boundingBox.size_x = box.xmax - box.xmin
            self.boundingBox.size_y = box.ymax - box.ymin
            self.boundingBox.center = Point(int((box.xmax + box.xmin)/2), int((box.ymax + box.ymin)/2), 0)

    def CropDepthImg(self, img, imgEncoding, boxCenter, boxWidth, boxHeight):
        if imgEncoding == "32FC1":
            imageHeight, imageWidth = img.shape
        else:
            imageHeight, imageWidth, a = img.shape

        boxWidth = max(abs(boxWidth*imageWidth), 2)
        boxHeight = max(abs(boxHeight*imageHeight), 2)

        x0 = max(int(boxCenter.x * imageWidth - boxWidth/2), 0)
        y0 = max(int(boxCenter.y * imageHeight - boxHeight/2), 0)

        xf = min(int(boxCenter.x * imageWidth + boxWidth/2), imageWidth)
        yf = min(int(boxCenter.y * imageHeight + boxHeight/2), imageHeight)

        rospy.loginfo(imageWidth, imageHeight)
        rospy.loginfo(x0, y0, xf, yf)

        cropped_img = img[y0:yf, x0:xf]
        return cropped_img

    def GetAverageDepth(self, depthImg):
        height, width = depthImg.shape
        npArray = depthImg[0:height, 0:width]

        rowMeans = np.array([])
        for row in npArray:
            rowMeans = np.append(rowMeans, np.mean(row))

        depthMean = np.mean(rowMeans)
        return depthMean

    ''' By using rule of three and considering the FOV of the camera:
            - Calculates the 3D point of a depth pixel '''
    def Get3dPointFromDepthPixel(self, pixelPoint, depth):

        # Constants
        maxAngle_x = self.camFov_horizontal/2
        maxAngle_y = self.camFov_vertical/2
        screenMax_x = 1.0
        screenMax_y = 1.0
        screenCenter_x = screenMax_x / 2.0
        screenCenter_y = screenMax_y / 2.0

        # Distances to screen center
        distanceToCenter_x = pixelPoint.x - screenCenter_x
        distanceToCenter_y = pixelPoint.y - screenCenter_y

        # Horizontal angle (xz plane)
        xz_angle_deg = maxAngle_x * distanceToCenter_x / screenCenter_x
        xz_angle_rad = radians(xz_angle_deg)
        
        # Vertical angle (yz plane)
        yz_angle_deg = maxAngle_y * distanceToCenter_y / screenCenter_y
        yz_angle_rad = radians(yz_angle_deg)

        # Coordinates
        num = depth
        denom = sqrt(1 + pow(tan(xz_angle_rad), 2) + pow(tan(yz_angle_rad), 2))
        z = num / denom
        x = z * tan(xz_angle_rad)
        y = z * tan(yz_angle_rad)

        # Corrections
        x = -x
        y = -y

        return Point(x, y, z)

    def XyzToZxy(self, point):
        return Point(point.z, point.x, point.y)

    def mainLoop(self):
        while rospy.is_shutdown() == False:
            self.loopRate.sleep()
            if self.newDepthImg == True: #and self.newBoundingBox == True:
                # rospy.loginfo("1 - new bb and depth")
                self.newDepthImg = False
                self.newBoundingBox = False
                cv_depthImg = self.cvBridge.imgmsg_to_cv2(self.msg_depthImg, "32FC1")

                try:
                    # rospy.loginfo("2 - attempting crop")
                    croppedDepthImg = self.CropDepthImg(
                        cv_depthImg,
                        "32FC1",
                        self.msg_boundingBox_center,
                        self.msg_boundingBox_size_x,
                        self.msg_boundingBox_size_y)
                    rospy.loginfo("3 - cropped")

                    if cv2.waitKey(5) & 0xFF == 27:
                        rospy.loginfo("E - breaking")
                        break
                    cv2.imshow("Cropped Depth", croppedDepthImg)
                    
                    rospy.loginfo("4 - averaging depth")
                    averageDepth = self.GetAverageDepth(croppedDepthImg)

                    self.msg_point = self.Get3dPointFromDepthPixel(
                        self.msg_boundingBox_center, 
                        averageDepth)
                    self.msg_point = self.XyzToZxy(self.msg_point)
                    self.pub_point.publish(self.msg_point)
                    rospy.loginfo(self.msg_point)

                except:
                    print("------------- Error in depth crop -------------")
                    continue

if __name__ == '__main__':
    DepthEstimatorNode(
        "/apollo/vision/depth_estimator/point", 
        "/camera/depth_registered/image_raw",
        "darknet_ros/bounding_boxes",
        43,
        57)
