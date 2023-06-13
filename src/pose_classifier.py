#!/usr/bin/env python3

# ROS
import rospy
## Message definitions gi
from std_msgs.msg import String
from geometry_msgs.msg import Point
from vision_msgs.msg import PointArray

class ClassifyPose():
    def __init__(self):
        # Messages
        self.msg_poseLandmarks = PointArray()
        self.msg_poseClass      = String()
        
        # To tell if there's a new msg
        self.newPoseLandmarks = False

        # Publishers and Subscribers
        self.sub_poseLandmarks = rospy.Subscriber(
                "/utbots/vision/person/pose/poseLandmarks", PointArray, 
                self.callback_poseLandmarks
        )
        self.pub_poseClass = rospy.Publisher(
             "/utbots/vision/person/pose/poseClass", String, queue_size=10
        )
        
        # ROS node
        rospy.init_node('pose_classifier', anonymous = True)
        self.mainLoop()

        # Time 
        self.loopRate = rospy.Rate(30)

# Callback
    def callback_poseLandmarks(self, msg):
        self.msg_poseLandmarks = msg
        self.newPoseLandmarks = True
    
# Pose Classification
    def ClassifyPose(self):
        print("falta coisa aqui")

# Publish
    def PublishPose(self):
        self.pub_poseClass.publish(self.msg_poseClass)

# Main
    def mainLoop(self):
        while rospy.is_shutdown() == False:
            self.loopRate.sleep()
            self.PublishPose()
            self.ClassifyPose()

if __name__ == '__main__':
    ClassifyPose()
