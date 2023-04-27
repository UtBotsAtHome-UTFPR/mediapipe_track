#!/usr/bin/env python

# python3 for Jetson Nano

# ROS
import rospy
# OpenCV
import cv2
from cv_bridge import CvBridge
# Message definitions
from std_msgs.msg import Int8MultiArray
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
# Math
from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
from math import pow, sqrt, tan, radians

class DomTorsoColor:
    def __init__(self, topic_torsoImg, topic_torsoColor):
        # Subscribe and Publish Topics
        self.topic_torsoImg = topic_torsoImg
        self.topic_torsoColor = topic_torsoColor

        # Control flag for a new image
        self.newImg = False

        # Messages
        self.msg_torsoImg            = None      # Image
        self.msg_torsoRGBlist        = Int8MultiArray()
        # self.msg_torsoRGBlist.data   = []

        # Subscribers and Publishers
        self.pub_torsoColor = rospy.Publisher(
            topic_torsoColor, Int8MultiArray, queue_size=1)
        self.sub_rgbImg = rospy.Subscriber(
            topic_torsoImg, Image, self.callback_torsoImg)
        self.sub_getPersonId = rospy.Subscriber(
        "/utbots/vision/lock/target/get_person_id",Bool, self.callback_getpersonid)

        self.meanColor = [0,0,0]
        self.maxDistRGB = [0,0,0]
        self.maxEucDist = 0
        self.iteration = 1
        self.get_person_id = False

        # ROS node
        rospy.init_node('dom_torso_color', anonymous=True)

        # Time
        self.loopRate = rospy.Rate(60)

        # Calls main loop
        self.mainLoop()

# Callbacks
    def callback_torsoImg(self, msg):
        self.msg_torsoImg = msg
        self.newImg = True
        # print("- RGB: new msg")

    def imgToCv(self, imgMsg):
        # Conversion to cv image
        bridge = CvBridge()
        cvImg = bridge.imgmsg_to_cv2(imgMsg, "8UC3")
        return cvImg

    def callback_getpersonid(self, msg):
        self.get_person_id = msg.data

    def get_dominant_color(self, image):
        k=4
        image_processing_size = None

        #resize image if new dims provided
        if image_processing_size is not None:
            image = cv2.resize(image, image_processing_size, 
                                interpolation = cv2.INTER_AREA)
        
        #reshape the image to be a list of pixels
        image = image.reshape((image.shape[0] * image.shape[1], 3))

        #cluster and assign labels to the pixels 
        clt = KMeans(n_clusters = k)
        labels = clt.fit_predict(image)

        #count labels to find most popular
        label_counts = Counter(labels)

        #subset out most popular centroid
        dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

        rgbList = list(dominant_color)

        # Convert from numpy.float64 to int
        for i in range(len(rgbList)):
            rgbList[i] = int(rgbList[i])

        return rgbList
    
    def saveColor(self, rgb):
        rospy.loginfo("Saving ID")
        if self.iteration >= 1:
            for i in range(len(rgb)):
                self.meanColor[i] = self.movingAverage(rgb[i], self.meanColor[i], self.iteration)
                self.maxDistRGB[i] = self.maxDifference(rgb[i], self.meanColor[i], self.maxDistRGB[i])
            self.maxEucDist = self.EucDist(rgb, self.meanColor)
        else: 
            for i in range(len(rgb)):
                self.meanColor[i] = rgb[i]
        self.iteration += 1

    def movingAverage(self, new_value, current_mean, iteration):
        return current_mean+(new_value-current_mean)/iteration
    
    def maxDifference(self, current, mean, max):
        dif = abs(current - mean)
        print(dif)
        if dif > max:
           max = dif
        return max

    def EucDist(self, x, y):
        return sqrt(pow(x[0]-y[0], 2)+pow(x[1]-y[1], 2)+pow(x[2]-y[2], 2))

    def mainLoop(self):

        while rospy.is_shutdown() == False:
            self.loopRate.sleep()

            if self.newImg == True:

                # Get torso colors
                rbgList = self.get_dominant_color(self.imgToCv(self.msg_torsoImg))
                if self.get_person_id == True:
                    self.saveColor(rbgList)
                    if self.iteration >= 5:
                        self.get_person_id = False
                        rospy.loginfo("Mean: {} Max Distance: {} Max Difference: {}".format(self.meanColor, self.maxEucDist, self.maxDistRGB))
                self.msg_torsoRGBlist.data = rbgList
                rospy.loginfo("{} {}".format(self.msg_torsoRGBlist.data, self.EucDist(rbgList, self.meanColor)))
                self.newImg = False

                # Publish torso color
                self.pub_torsoColor.publish(self.msg_torsoRGBlist)

                # Display color
                height=512
                width=512
                blank_image = np.zeros((height,width,3), np.uint8)
                blank_image[:]=rbgList
                cv2.imshow('Torso Color', blank_image)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()

if __name__ == "__main__":
    DomTorsoColor(
        "/utbots/vision/lock/target/rgb",
        "/utbots/vision/lock/target/torsoColor")