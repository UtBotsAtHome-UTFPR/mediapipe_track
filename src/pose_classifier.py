#!/usr/bin/env python3

# ROS
import rospy

# Message definitions 
from std_msgs.msg import String
from geometry_msgs.msg import Point
from vision_msgs.msg import Skeleton2d

# Math
import math


class ClassifyPose():
    def __init__(self):
        # Messages
        self.msg_poseLandmarks = Skeleton2d()
        self.msg_poseClass      = String()
        
        # To tell if there's a new msg
        self.newPoseLandmarks = False

        # Publishers and Subscribers
        self.sub_poseLandmarks = rospy.Subscriber(
                "/utbots/vision/person/pose/poseLandmarks", Skeleton2d, 
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
        poseClass = 'Unknown'
        
        # Calculate key angles
        left_elbow_angle = self.CalculateAngle(self.msg_poseLandmarks.LEFT_SHOULDER,
                                          self.msg_poseLandmarks.LEFT_ELBOW,
                                          self.msg_poseLandmarks.LEFT_WRIST)
    
        # Get the angle between the right shoulder, elbow and wrist points. 
        right_elbow_angle = self.CalculateAngle(self.msg_poseLandmarks.RIGHT_SHOULDER,
                                           self.msg_poseLandmarks.RIGHT_ELBOW,
                                           self.msg_poseLandmarks.RIGHT_WRIST)   
    
        # Get the angle between the left elbow, shoulder and hip points. 
        left_shoulder_angle = self.CalculateAngle(self.msg_poseLandmarks.LEFT_ELBOW,
                                             self.msg_poseLandmarks.LEFT_SHOULDER,
                                             self.msg_poseLandmarks.LEFT_HIP)

        # Get the angle between the right hip, shoulder and elbow points. 
        right_shoulder_angle = self.CalculateAngle(self.msg_poseLandmarks.RIGHT_HIP,
                                              self.msg_poseLandmarks.RIGHT_SHOULDER,
                                              self.msg_poseLandmarks.RIGHT_ELBOW)

        # Get the angle between the left hip, knee and ankle points. 
        left_knee_angle = self.CalculateAngle(self.msg_poseLandmarks.LEFT_HIP,
                                         self.msg_poseLandmarks.LEFT_KNEE,
                                         self.msg_poseLandmarks.LEFT_ANKLE)

        # Get the angle between the right hip, knee and ankle points 
        right_knee_angle = self.CalculateAngle(self.msg_poseLandmarks.RIGHT_HIP,
                                          self.msg_poseLandmarks.RIGHT_KNEE,
                                          self.msg_poseLandmarks.RIGHT_ANKLE)

        if self.Standing(left_knee_angle, right_knee_angle) == True:
            poseClass = 'Standing'

        return poseClass

    def Standing(self, left_knee_angle, right_knee_angle):
        
        # Check if one of the legs is straight 
        if(left_knee_angle > 165 and left_knee_angle < 195):
            straight_leg = True

        elif(right_knee_angle > 165 and right_knee_angle < 195):
            straight_leg = True
        
        if(straight_leg and 
           self.msg_poseLandmarks.NOSE.y < self.msg_poseLandmarks.RIGHT_SHOULDER.y and
           self.msg_poseLandmarks.RIGHT_SHOULDER.y < self.msg_poseLandmarks.RIGHT_HIP.y):
            return True
        else:
            return False
        

# Calculates the angle between three diferent points
    def sCalculateAngle(self, point1, point2, point3):
        # Calculate the angle
        angle = math.degrees(math.atan2(point3.y - point2.y, point3.x - point2.x) - 
                             math.atan2(point1.y - point2.y, point1.x - point2.x))

        # Check if the angle is less than zero.
        if angle < 0:
            # Add 360 to the found angle.
            angle += 360
    
        # Return the calculated angle.
        return angle

# Publish
    def PublishPose(self):
        self.pub_poseClass.publish(self.msg_poseClass)

# Main
    def mainLoop(self):
        while rospy.is_shutdown() == False:
            self.loopRate.sleep()
            
            # Classifies a new pose only if there are new pose landamarks 
            if self.newPoseLandmarks == True:
                self.newPoseLandmarks = False

                self.msg_poseClass = self.ClassifyPose()
            
            self.PublishPose()
            

if __name__ == '__main__':
    ClassifyPose()
