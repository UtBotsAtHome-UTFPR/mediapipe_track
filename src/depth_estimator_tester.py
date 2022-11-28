#!/usr/bin/env python
from tkinter import mainloop
import rospy
from sensor_msgs.msg import Image
from vision_msgs.msg import BoundingBox2D
from geometry_msgs.msg import Pose2D

class Node():
    def __init__(self):
        # ROS node
        rospy.init_node('depth_estimator_tester', anonymous=True)

        # Subscribed depth img
        self.topic_InDepthImg = "/camera/depth_registered/image_raw"
        self.msg_InDepthImg = None
        self.sub_depthImg = rospy.Subscriber(
            self.topic_InDepthImg,
            Image,
            self.callback_InDepthImg,
            queue_size=1)

        # Published depth img
        self.topic_depthImg = "/apollo/vision/depth_estimator/depth"
        self.msg_depthImg = None
        self.pub_depthImg = rospy.Publisher(
            self.topic_depthImg, 
            Image, 
            queue_size = 1)

        # Published bounding box
        self.topic_boundingBox = "/apollo/vision/depth_estimator/box"
        self.msg_boundingBox = BoundingBox2D(Pose2D(0.5, 0.5, 0), 1.0, 1.0)
        self.pub_boundingBox = rospy.Publisher(
            self.topic_boundingBox,
            BoundingBox2D,
            queue_size = 1)

        self.loopRate = rospy.Rate(30)

        rospy.loginfo("Looping...")
        self.mainLoop()

    def callback_InDepthImg(self, msg):
        self.msg_InDepthImg = msg
        rospy.loginfo("Callback depth img")

    def mainLoop(self):
        while rospy.is_shutdown() == False:
            self.loopRate.sleep()
            
            if self.msg_InDepthImg != None:
                rospy.loginfo("Publishing depth image and box...")
                self.msg_depthImg = self.msg_InDepthImg
                self.pub_depthImg.publish(self.msg_depthImg)
                self.pub_boundingBox.publish(self.msg_boundingBox)

if __name__ == "__main__":
    Node()