#!/usr/bin/env python3

import os

# Mediapipe imports
import cv2
import mediapipe as mp

# Image processing
from cv_bridge import CvBridge
from collections import Counter

# ROS
import rospy
## Message definitions
from std_msgs.msg import String
from std_msgs.msg import Bool
from geometry_msgs.msg import PointStamped, Point, TransformStamped
from sensor_msgs.msg import Image
from vision_msgs.msg import PointArray
## Transformation tree
from tf.msg import tfMessage
from geometry_msgs.msg import TransformStamped
from tf.transformations import quaternion_from_euler

# Math 
import numpy as np
from math import pow, sqrt, tan, radians
import random

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
            "/humans/bodies/status", String, queue_size=10)
        self.pub_targetPoint = rospy.Publisher(
            "/humans/bodies/torsoPoint", PointStamped, queue_size=10)
        self.pub_targetCroppedRgbTorso = rospy.Publisher(
            "/humans/bodies/croppedTorso/rgb", Image, queue_size=10)
        self.pub_targetCroppedDepthTorso = rospy.Publisher(
            "/humans/bodies/croppedTorso/depth", Image, queue_size=10)
        self.pub_targetSkeletonImg = rospy.Publisher(
            "/humans/bodies/skeletonImg", Image, queue_size=10)
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
        for lmark in landmarks:
            landmarks.append(self.PointByLmarks(landmarks[lmark]))
        self.msg_poseLandmarks.points = landmarks

## Torso
    def GetTorsoPoints(self):
        return [self.PointByLmarks(self.trackedBody.shoulder.landmarks_list[1]),
                self.PointByLmarks(self.trackedBody.shoulder.landmarks_list[0]),
                self.PointByLmarks(self.trackedBody.hip.landmarks_list[1]),
                self.PointByLmarks(self.trackedBody.hip.landmarks_list[0])]

    def PointByLmarks(self, landmark):
        return Point(landmark.x, landmark.y, landmark.z)

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
        num = depth / 1000
        denom = sqrt(1 + pow(tan(xz_angle_rad), 2) + pow(tan(yz_angle_rad), 2))
        z = (num / denom)
        x = z * tan(xz_angle_rad)
        y = z * tan(yz_angle_rad)

        # Corrections
        x = -x
        y = -y

        # print("depth: {}".format(depth))
        # print("distancesToCenter: ({}, {})".format(distanceToCenter_x, distanceToCenter_y))
        # print("angles: ({}, {})".format(xz_angle_deg, yz_angle_deg))
        # print("xyz: ({}, {}, {})".format(x, y, z))

        return Point(x, y, z)

    ''' Transforms the mpipe coordinate format to tf tree coordinate format'''
    def XyzToZxy(self, point):
        return Point(point.z, point.x, point.y)   

    def ExtractDepthPoint(self, coordinate, depth_frame):
        depthPoint = self.Get3dPointFromDepthPixel(coordinate, depth_frame)
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

    def LandmarksProcessing(self):  # Return info: 1 -> break; 0 -> continue
        if self.newRgbImg == True:
            self.newRgbImg = False

            cv_rgbImg, poseResults = self.ProcessImg()
            self.DrawLandmarks(cv_rgbImg, poseResults)
            self.msg_targetSkeletonImg = self.cvBridge.cv2_to_imgmsg(cv_rgbImg)

            # Pose WORLD Landmarks, otherwise the data format does not represent metric real distances
            if poseResults.pose_world_landmarks:

                self.DefineBodyStructure(poseResults.pose_landmarks.landmark)    
                torsoPoints = self.GetTorsoPoints()
                torsoCenter = self.GetPointsMean(torsoPoints)
                self.SetLandmarkPoints(poseResults.pose_landmarks.landmark)

                try:
                    croppedRgbImg = self.CropTorsoImg(cv_rgbImg, "passthrough", torsoPoints, torsoCenter)
                    self.msg_targetCroppedRgbTorso = self.cvBridge.cv2_to_imgmsg(croppedRgbImg)
                except:
                    rospy.loginfo("------------- Error in RGB crop -------------")
                    return 0

                if self.newDepthImg == True:
                    cv_depthImg = self.cvBridge.imgmsg_to_cv2(self.msg_depthImg, "32FC1")
                    try:
                        croppedDepthImg = self.CropTorsoImg(cv_depthImg, "32FC1", torsoPoints, torsoCenter)
                        self.msg_targetCroppedDepthTorso = self.cvBridge.cv2_to_imgmsg(croppedDepthImg)
                        # torsoCenter3d = self.Get3dPointFromDepthPixel(torsoCenter, self.GetTorsoDistance(croppedDepthImg))
                        # torsoCenter3d = self.XyzToZxy(torsoCenter3d)
                        torsoCenter3d = self.ExtractDepthPoint(torsoCenter, self.GetTorsoDistance(croppedDepthImg))
                        # self.msg_targetPoint = Point(self.GetTorsoDistance(croppedDepthImg), 0, 0)
                        self.msg_targetPoint.point = torsoCenter3d
                        self.msg_targetStatus = "Located"
                    except:
                        rospy.loginfo("------------- Error in depth crop -------------")
                        return 0                
                # Nothing detected...
                else:
                    t_now = rospy.get_time()
                    if (t_now - self.t_last > self.t_timeout and self.msg_targetStatus != "?"):
                        self.t_last = t_now
                        self.msg_targetPoint.point = Point(0, 0, 0)
                        self.msg_targetStatus = "?"

# Nodes Publish
    def PublishEverything(self):
        self.pub_targetCroppedRgbTorso.publish(self.msg_targetCroppedRgbTorso)
        self.pub_targetCroppedDepthTorso.publish(self.msg_targetCroppedDepthTorso)
        self.pub_targetStatus.publish(self.msg_targetStatus)
        self.pub_targetSkeletonImg.publish(self.msg_targetSkeletonImg)
        self.pub_poseLandmarks.publish(self.msg_poseLandmarks)
        # self.SetupTfMsg(self.msg_targetPoint.point.x, self.msg_targetPoint.point.y, self.msg_targetPoint.point.z)

# Main
    def mainLoop(self):
        while rospy.is_shutdown() == False:
            self.loopRate.sleep()
            self.PublishEverything()

            # print("\nTARGET")
            # print(" - status: {}".format(self.msg_targetStatus))
            # print(" - xyz: ({}, {}, {})".format(
            #     self.msg_targetPoint.point.x, 
            #     self.msg_targetPoint.point.y, 
            #     self.msg_targetPoint.point.z))

            # Coordinates every method that processes pose landmarks
            lmark_processing = self.LandmarksProcessing()
            # Errors in the landmark processing can alter the loop flows
            if lmark_processing == -1:
                break
            elif lmark_processing == 0:
                continue
              
if __name__ == "__main__":
    LockPose(
        "/camera/rgb/image_raw",
        "/camera/depth_registered/image_raw",
        43,
        57)