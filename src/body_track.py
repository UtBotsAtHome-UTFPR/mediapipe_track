#!/usr/bin/env python3

# Mediapipe imports
import cv2
import mediapipe as mp

# Image processing
from cv_bridge import CvBridge

# ROS
import rospy
## Message definitions
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, Point, TransformStamped
from sensor_msgs.msg import Image
from vision_msgs.msg import PointArray
## Transformation tree
from tf.msg import tfMessage
from geometry_msgs.msg import TransformStamped

class Person():
    def __init__(self):
        self.existsID = False

        # Body parts objects
        self.shoulder = BodyPart("shoulder")
        self.hip = BodyPart("hip")
        self.torsoHeight = BodyPart("torso")
        self.rightLeg = BodyPart("right_leg")
        self.leftLeg = BodyPart("left_leg")
        self.rightArm = BodyPart("right_arm")
        self.leftArm = BodyPart("left_arm")

        # Body parts list
        self.bodyParts = [self.shoulder, self.hip, self.rightLeg, self.leftLeg, self.rightArm, self.leftArm]

        # Body visibility flag
        self.bodyVisible = False

    def ChkBodyVisib(self):
        for limb in self.bodyParts:
            if(limb.isVisible == False):
                rospy.loginfo(limb.name + " not visible")
                self.bodyVisible = False
                return False
        self.bodyVisible = True
        return True

class BodyPart():
    def __init__(self, name):
        self.name = name

        self.landmarks_list = None
        self.MIN_VISIBILITY = 0.5
        self.isVisible = False

    def SetLandmarkList(self, lmark_list):
        self.landmarks_list = lmark_list
        self.isVisible = self.ChkVisib(lmark_list)

    def ChkVisib(self, lmark_list):
        for i in range(len(lmark_list)):
            if lmark_list[i].visibility < self.MIN_VISIBILITY:
                # rospy.loginfo("WARN: Points not visible")
                return False
        return True

class LockPose():
    def __init__(self, topic_rgbImg, topic_depthImg, camFov_vertical, camFov_horizontal):
        # Image FOV for trig calculations
        self.camFov_vertical = camFov_vertical
        self.camFov_horizontal = camFov_horizontal

        # Messages
        self.msg_tfStamped              = TransformStamped()
        self.msg_targetStatus           = "?" # String
        self.msg_targetCroppedRgbTorso    = Image()
        self.msg_targetCroppedDepthTorso  = Image()
        self.msg_targetSkeletonImg      = Image()
        self.msg_rgbImg                 = None      # Image
        self.msg_depthImg               = None      # Image
        self.msg_poseLandmarks          = PointArray()
        self.msg_targetStatus                = "?"
        self.pub_targetStatus = rospy.Publisher(
        "/utbots/vision/lock/status", String, queue_size=10)


        # To tell if there's a new msg
        self.newRgbImg = False
        self.newDepthImg = False

        # Publishers and Subscribers
        self.sub_rgbImg = rospy.Subscriber(
            topic_rgbImg, Image, self.callback_rgbImg)
        self.sub_depthImg = rospy.Subscriber(
            topic_depthImg, Image, self.callback_depthImg)

        self.pub_tf = rospy.Publisher(
            "/tf", tfMessage, queue_size=1)
        self.pub_targetStatus = rospy.Publisher(
            "/utbots/vision/lock/status", String, queue_size=10)
        self.pub_targetPoint = rospy.Publisher(
            "/utbots/vision/lock/torsoPoint", PointStamped, queue_size=10)
        self.pub_targetSkeletonImg = rospy.Publisher(
            "/utbots/vision/lock/skeletonImg", Image, queue_size=10)
        self.pub_poseLandmarks = rospy.Publisher(
            "/utbots/vision/lock/poseLandmarks", PointArray, queue_size=10)

        # ROS node
        rospy.init_node('locker_human', anonymous=True)

        # Time
        self.loopRate = rospy.Rate(30)
        self.t_last = 0.0  # sec
        self.t_timeout = 0.250  # sec

        # Cv
        self.cvBridge = CvBridge()

        # Mediapipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose

        # Person
        self.trackedBody = Person()

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
        # print("- RGB: new msg")

    def callback_depthImg(self, msg):
        self.msg_depthImg = msg
        self.newDepthImg = True
        # print("- Depth: new msg")
    
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

        # Returns
        return cvImg, poseResults

    def DrawLandmarks(self, cv_rgbImg, poseResults):
        self.mp_drawing.draw_landmarks(
            cv_rgbImg,
            poseResults.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

# Body data processing
## Limbs distances
    def DefineBodyStructure(self, landmark):
        # Evaluated landmark points
        ## Elbows
        rElbow = landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        lElbow = landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW] 
        ## Wrists
        rWrist = landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        lWrist = landmark[self.mp_pose.PoseLandmark.LEFT_WRIST] 
        ## Hip
        rHip = landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        lHip = landmark[self.mp_pose.PoseLandmark.LEFT_HIP] 
        ## Shoulder
        rShoulder = landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lShoulder = landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        ## Knees
        rKnee = landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        lKnee = landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
        ## Ankles
        rAnkle = landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        lAnkle = landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]

        self.trackedBody.shoulder.SetLandmarkList([lShoulder, rShoulder])
        self.trackedBody.hip.SetLandmarkList([lHip, rHip])
        self.trackedBody.torsoHeight.SetLandmarkList([lHip, rHip, lShoulder, rShoulder])
        self.trackedBody.rightLeg.SetLandmarkList([rHip, rKnee, rAnkle])
        self.trackedBody.leftLeg.SetLandmarkList([lHip, lKnee, lAnkle])
        self.trackedBody.rightArm.SetLandmarkList([rShoulder, rElbow, rWrist])
        self.trackedBody.leftArm.SetLandmarkList([lShoulder, lElbow, lWrist])

    def SetLandmarkPoints(self, landmark):
        landmarks = []
        for lmark in self.mp_pose.PoseLandmark:
            landmarks.append(self.PointByLmarks(landmark[lmark]))
        self.msg_poseLandmarks.points = landmarks

    def PointByLmarks(self, landmark):
        return Point(landmark.x, landmark.y, landmark.z)

# Nodes Publish
    def PublishEverything(self):
        self.pub_targetStatus.publish(self.msg_targetStatus)
        self.pub_targetSkeletonImg.publish(self.msg_targetSkeletonImg)
        self.pub_poseLandmarks.publish(self.msg_poseLandmarks)
        self.pub_targetStatus.publish(self.msg_targetStatus)

# Main
    def mainLoop(self):
        while rospy.is_shutdown() == False:
            self.loopRate.sleep()
            self.PublishEverything()

            if self.newRgbImg == True:
                self.newRgbImg = False

                cv_rgbImg, poseResults = self.ProcessImg()
                self.DrawLandmarks(cv_rgbImg, poseResults)
                self.msg_targetSkeletonImg = self.cvBridge.cv2_to_imgmsg(cv_rgbImg)

                # Pose WORLD Landmarks, otherwise the data format does not represent metric real distances
                if poseResults.pose_world_landmarks:
                    self.DefineBodyStructure(poseResults.pose_landmarks.landmark)    
                    self.SetLandmarkPoints(poseResults.pose_landmarks.landmark)
                    self.msg_targetStatus = "Detected"
                else:
                    self.msg_targetStatus = "Not Detected"
              
if __name__ == "__main__":
    LockPose(
        "/camera/rgb/image_raw",
        "/camera/depth_registered/image_raw",
        43,
        57)
