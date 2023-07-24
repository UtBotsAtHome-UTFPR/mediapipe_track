#!/usr/bin/env python3

# ROS implementation of the Mediapipe Pose solution up to the 2022 libraries
# Inputs the sensor_msgs/Image message to feed the landmarks processing, outputs the skeletonImage, the detection status and the landmarks list

# Mediapipe imports
import cv2
import mediapipe as mp

# Image processing
from cv_bridge import CvBridge

# ROS
import rospy
## Message definitions
from std_msgs.msg import String
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from vision_msgs.msg import Skeleton2d

class BodyPose():
    def __init__(self, topic_rgbImg):

        # Messages
        self.msg_rgbImg                 = None      # Image
        self.msg_targetStatus           = "Not Detected" # String
        self.msg_targetSkeletonImg      = Image()
        self.msg_poseLandmarks          = Skeleton2d()
        self.msg_poseWorldLandmarks     = Skeleton2d()

        # To tell if there's a new msg
        self.newRgbImg = False

        # Publishers and Subscribers
        self.sub_rgbImg = rospy.Subscriber(
            topic_rgbImg, Image, self.callback_rgbImg)

        self.pub_targetStatus = rospy.Publisher(
            "pose/status", String, queue_size=10)  
        self.pub_targetSkeletonImg = rospy.Publisher(
            "pose/skeletonImg", Image, queue_size=10)
        self.pub_poseLandmarks = rospy.Publisher(
            "pose/poseLandmarks", Skeleton2d, queue_size=10)
        self.pub_poseWorldLandmarks = rospy.Publisher(
            "pose/poseWorldLandmarks", Skeleton2d, queue_size=10)

        # ROS node
        rospy.init_node('body_pose', anonymous=True)

        # Time
        self.loopRate = rospy.Rate(30)

        # Cv
        self.cvBridge = CvBridge()

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
        self.newRgbImg = True
    
# Basic MediaPipe Pose methods
    def ProcessImg(self):
        # Conversion to cv image
        cvImg = self.cvBridge.imgmsg_to_cv2(self.msg_rgbImg, "bgr8")

        # Not writeable passes by reference (better performance)
        cvImg.flags.writeable = False

        # Converts BGR to RGB
        cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)

        # Image processing
        poseResults = self.pose.process(cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB))

        # To draw the hand annotations on the image
        cvImg.flags.writeable = True

        # Back to BGR
        cvImg = cv2.cvtColor(cvImg, cv2.COLOR_RGB2BGR)

        return cvImg, poseResults

    def DrawLandmarks(self, cv_rgbImg, poseResults):
        self.mp_drawing.draw_landmarks(
            cv_rgbImg,
            poseResults.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

# Body data processing
    def SetLandmarkPoints(self, landmark):
        landmarks = []
        for lmark in self.mp_pose.PoseLandmark:
            landmarks.append(self.PointByLmarks(landmark[lmark]))
        return landmarks

    def PointByLmarks(self, landmark):
        return Point(landmark.x, landmark.y, landmark.z)

# Nodes Publish
    def PublishEverything(self):
        self.pub_targetSkeletonImg.publish(self.msg_targetSkeletonImg)
        self.pub_poseLandmarks.publish(self.msg_poseLandmarks)
        self.pub_poseWorldLandmarks.publish(self.msg_poseWorldLandmarks)
        self.pub_targetStatus.publish(self.msg_targetStatus)

# Main
    def mainLoop(self):
        while rospy.is_shutdown() == False:
            self.loopRate.sleep()

            # Gathers new landmark information only if there is a new image
            if self.newRgbImg == True:
                self.newRgbImg = False

                # Process the results
                cv_rgbImg, poseResults = self.ProcessImg()

                # Draws the skeleton image
                self.DrawLandmarks(cv_rgbImg, poseResults)
                self.msg_targetSkeletonImg = self.cvBridge.cv2_to_imgmsg(cv_rgbImg)

                # Processes pose results 
                if poseResults.pose_landmarks:
                    # Pose Landmarks are normalized image coordinates
                    self.msg_poseLandmarks.points = self.SetLandmarkPoints(poseResults.pose_landmarks.landmark)
                    # Pose WORLD Landmarks represent metric coordinates in the 3d space from the camera reference
                    self.msg_poseWorldLandmarks.points = self.SetLandmarkPoints(poseResults.pose_world_landmarks.landmark)
                    self.msg_targetStatus = "Detected"
                else:
                    self.msg_targetStatus = "Not Detected"

            self.PublishEverything()
    
if __name__ == "__main__":
    BodyPose(
        "/camera/rgb/image_raw")
