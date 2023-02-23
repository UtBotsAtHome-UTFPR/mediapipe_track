#!/usr/bin/env python

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
## Transformation tree
from tf.msg import tfMessage
from geometry_msgs.msg import TransformStamped
from tf.transformations import quaternion_from_euler

# Math 
import numpy as np
from math import pow, sqrt, tan, radians

class Person():
    def __init__(self, person_id):
        self.id = person_id
        self.existsID = False

        # Body parts objects
        self.shoulder = BodyPart("shoulder")
        self.hip = BodyPart("hip")
        self.torsoHeight = BodyPart("torso")
        self.rightLeg = BodyPart("right_leg")
        self.leftLeg = BodyPart("left_leg")

        # Body parts list
        self.bodyParts = [self.shoulder, self.hip, self.torsoHeight, self.rightLeg, self.leftLeg]

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
    
    def SetBodyPartValue(self, instance, value):
        if(instance.isVisible):
            instance.SetValue(value)
        else:
            self.bodyVisible = False

    def SetTorsoValue(self, torsoLeft, torsoRight):
        if(self.shoulder.isVisible and self.hip.isVisible):
            self.torsoHeight.SetValue((torsoLeft+torsoRight)/2)
        else:
            self.bodyVisible = False
        
    def RegisterPersonID(self):
        if(self.ChkBodyVisib()):
            for limb in self.bodyParts:
                print(limb.name + ":")
                limb.Statistics()
        else:
            rospy.loginfo("WARN: ALL body must be visible for operator registration.\nReposition yourself to finish the registration process")
    
    def ResetIterations(self):
        for limb in self.bodyParts:
            limb.ResetIteration()

    def CheckNSamples(self, nSamples):
        sum = 0
        for limb in self.bodyParts:
            sum += limb.iteration
        if(sum < nSamples):
            return False
        else:
            self.ResetIterations()
            return True

    def EvaluateSimilarity(self):
        for limb in self.bodyParts:
            limb.CalcDifference()
            print(limb.name + ": " + str(limb.difference))
            if(limb.difference > 0.1):
                rospy.loginfo("Is NOT the operator")
                return
        rospy.loginfo("Is the operator")


class BodyPart():
    def __init__(self, name):
        self.name = name
        self.value = 0

        self.landmarks_list = None
        self.MIN_VISIBILITY = 0.5
        self.isVisible = False

        self.iteration = 0
        self.meanSize = 0
        self.minSize = 0
        self.maxSize = 0
        self.sizeVariance = 0

        self.difference = 0

    def SetLandmarkList(self, lmark_list):
        self.landmarks_list = lmark_list
        self.isVisible = self.ChkVisib(lmark_list)

    def SetValue(self, value):
        self.value = value

    def CalcDifference(self):
        self.difference =  abs(self.value - self.meanSize)

    def ChkVisib(self, lmark_list):
        for i in range(len(lmark_list)):
            if lmark_list[i].visibility < self.MIN_VISIBILITY:
                # rospy.loginfo("WARN: Points not visible")
                return False
        return True

    def ResetIteration(self):
        self.iteration = 0

    def Statistics(self):
        if self.iteration < 1:
            firstValue = self.value
            self.maxSize = firstValue
            self.minSize = firstValue
        elif self.iteration < 2:
            self.meanSize = (self.meanSize+self.value)/2
        else:
            # Moving Average of Streaming Data
            self.meanSize = self.meanSize+(self.value-self.meanSize)/self.iteration
            ##relevance = 1/self.iteration
            ##self.meanSize = (self.meanSize+self.value*relevance)/(1+relevance)

            # Maximum Value
            # if self.value > self.maxSize:
            #     self.maxSize = self.value

            # Maximum Value
            # if self.value < self.minSize:
            #     self.minSize = self.value
            
            # Variance
            # self.varianceSize = self.maxSize - self.minSize

            # print("Iteration: {}\nGMean: {}\nGMin: {}\nGMax: {}\nGVar: {}".format(self.iteration, 
            # self.meanSize,self.minSize,self.maxSize,self.varianceSize))
            print("Iteration: {}, Mean: {}".format(self.iteration, self.meanSize))

        # Iteration
        self.iteration+=1

class LockPose():
    def __init__(self, topic_rgbImg, topic_depthImg, camFov_vertical, camFov_horizontal):
        # Image FOV for trig calculations
        self.camFov_vertical = camFov_vertical
        self.camFov_horizontal = camFov_horizontal

        # Messages
        self.msg_tfStamped              = TransformStamped()
        self.msg_targetStatus           = "?" # String
        self.msg_targetPoint            = PointStamped()   # Point
        self.msg_targetPoint.header.frame_id = "target"
        self.msg_targetCroppedRgbImg    = Image()
        self.msg_targetCroppedDepthImg  = Image()
        self.msg_targetSkeletonImg      = Image()
        self.msg_rgbImg                 = None      # Image
        self.msg_depthImg               = None      # Image
        self.get_person_id              = False     # Bool

        # To tell if there's a new msg
        self.newRgbImg = False
        self.newDepthImg = False

        # Publishers and Subscribers
        self.pub_tf = rospy.Publisher(
            "/tf", tfMessage, queue_size=1)
        self.pub_targetStatus = rospy.Publisher(
            "/utbots/vision/lock/target/status", String, queue_size=10)
        self.pub_targetPoint = rospy.Publisher(
            "/utbots/vision/lock/target/point", PointStamped, queue_size=10)
        self.pub_targetCroppedRgbImg = rospy.Publisher(
            "/utbots/vision/lock/target/rgb", Image, queue_size=10)
        self.pub_targetCroppedDepthImg = rospy.Publisher(
            "/utbots/vision/lock/target/depth", Image, queue_size=10)
        self.pub_targetSkeletonImg = rospy.Publisher(
            "/utbots/vision/lock/target/skeletonImage", Image, queue_size=10)
        self.sub_rgbImg = rospy.Subscriber(
            topic_rgbImg, Image, self.callback_rgbImg)
        self.sub_depthImg = rospy.Subscriber(
            topic_depthImg, Image, self.callback_depthImg)
        self.sub_getPersonId = rospy.Subscriber(
            "/utbots/vision/lock/target/get_person_id",Bool, self.callback_getpersonid)

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
        self.operator = Person("Operator")
        self.newID = False

        # Calls main loop
        self.pose = self.mp_pose.Pose(
            # Pose Configurations
            min_detection_confidence=0.75,
            min_tracking_confidence=0.9)
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
    
    def callback_getpersonid(self, msg):
        self.get_person_id = msg.data
        if msg == True:
            self.newID = True

# Basic MediaPipe Pose methods
    def ProcessImg(self, msg_img):
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
    def LimbsSizes(self, landmark, get_id):
        # Evaluated landmark points
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

        self.operator.shoulder.SetLandmarkList([lShoulder, rShoulder])
        self.operator.hip.SetLandmarkList([lHip, rHip])
        self.operator.torsoHeight.SetLandmarkList([lHip, rHip, lShoulder, rShoulder])
        self.operator.rightLeg.SetLandmarkList([rHip, rKnee, rAnkle])
        self.operator.leftLeg.SetLandmarkList([rHip, rKnee, rAnkle])

        os.system("clear")

        self.operator.SetBodyPartValue(self.operator.shoulder, self.EucDist(rShoulder, lShoulder))

        self.operator.SetBodyPartValue(self.operator.hip, self.EucDist(rHip, lHip))  

        self.operator.SetTorsoValue(self.EucDist(lHip, lShoulder), self.EucDist(rHip, rShoulder))

        self.operator.SetBodyPartValue(self.operator.rightLeg, self.EucDist(rHip, rKnee)+self.EucDist(rKnee,rAnkle))

        self.operator.SetBodyPartValue(self.operator.leftLeg, self.EucDist(lHip, lKnee)+self.EucDist(lKnee,lAnkle))

        if(get_id == True):
            rospy.loginfo("REGISTERING OPERATOR ID")
            self.operator.RegisterPersonID()
        elif(self.operator.existsID):   
            rospy.loginfo("EVALUATING OPERATOR SIMILARITY")
            self.operator.EvaluateSimilarity()
        else:
            rospy.loginfo("NO OPERATOR REGISTERED YET")

## Torso
    ''' Gets points for torso (shoulders and hips) '''
    def GetTorsoPoints(self, landmark):
        rightShoulder = Point(
            landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x, 
            landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y, 
            landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].z)
        leftShoulder = Point(
            landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x, 
            landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y, 
            landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].z)
        rightHip = Point(
            landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].x, 
            landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].y, 
            landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].z)
        leftHip = Point(
            landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x, 
            landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y, 
            landmark[self.mp_pose.PoseLandmark.LEFT_HIP].z)
        return [rightShoulder, leftShoulder, rightHip, leftHip]


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

    def EucDist(self, pointA, pointB):
        # print("A: ({:.2f}, {:.2f}, {:.2f})\nB: ({:.2f}, {:.2f}, {:.2f})".format(pointA.x,pointA.y,pointA.z,pointB.x,pointB.y,pointB.z))
        return sqrt(pow(pointA.x-pointB.x, 2)+pow(pointA.y-pointB.y, 2)+pow(pointA.z-pointB.z, 2))

    def MeanPixelColor(self, image):
        size = image.shape
        height_elem = 0
        width_elem = 0
        while height_elem < size[0]:
            while width_elem < size[1]:
                if height_elem == 0 and width_elem == 0:
                    (b,g,r) = image[height_elem][width_elem]
                else: 
                    (b,g,r) = (b,g,r)+(image[height_elem][width_elem]-(b,g,r))/(height_elem+width_elem)              
                width_elem+=1
            height_elem+=1
        print ("Mean Pixel Color - Red: {}, Green: {}, Blue: {}".format(r, g, b))
        return (r,g,b)


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

            cv_rgbImg, poseResults = self.ProcessImg(self.msg_rgbImg)
            self.DrawLandmarks(cv_rgbImg, poseResults)
            self.msg_targetSkeletonImg = self.cvBridge.cv2_to_imgmsg(cv_rgbImg)
            # cv2.imshow('MediaPipe Pose', cv_rgbImg)
            if cv2.waitKey(5) & 0xFF == 27:
                return -1

            # Pose WORLD Landmarks, otherwise the data format does not represent metric real distances
            if poseResults.pose_world_landmarks:

                self.LimbsSizes(poseResults.pose_world_landmarks.landmark, self.get_person_id)
                torsoPoints = self.GetTorsoPoints(poseResults.pose_landmarks.landmark)
                torsoCenter = self.GetPointsMean(torsoPoints)

                try:
                    croppedRgbImg = self.CropTorsoImg(cv_rgbImg, "passthrough", torsoPoints, torsoCenter)
                    self.msg_targetCroppedRgbImg = self.cvBridge.cv2_to_imgmsg(croppedRgbImg)
                    cv2.imshow("Cropped RGB", croppedRgbImg)
                except:
                    print("------------- Error in RGB crop -------------")
                    return 0

                if self.newDepthImg == True:
                    cv_depthImg = self.cvBridge.imgmsg_to_cv2(self.msg_depthImg, "32FC1")
                    cv2.imshow("depth Img", cv_depthImg)
                    try:
                        croppedDepthImg = self.CropTorsoImg(cv_depthImg, "32FC1", torsoPoints, torsoCenter)
                        self.msg_targetCroppedDepthImg = self.cvBridge.cv2_to_imgmsg(croppedDepthImg)
                        # cv2.imshow("Cropped Depth", croppedDepthImg)
                        torsoCenter3d = self.Get3dPointFromDepthPixel(torsoCenter, self.GetTorsoDistance(croppedDepthImg))
                        torsoCenter3d = self.XyzToZxy(torsoCenter3d)
                        # self.msg_targetPoint = Point(self.GetTorsoDistance(croppedDepthImg), 0, 0)
                        self.msg_targetPoint.point = torsoCenter3d
                        self.msg_targetStatus = "Located"
                    except:
                        print("------------- Error in depth crop -------------")
                        return 0                
                # Nothing detected...
                else:
                    t_now = rospy.get_time()
                    if (t_now - self.t_last > self.t_timeout and self.msg_targetStatus != "?"):
                        self.t_last = t_now
                        self.msg_targetPoint.point = Point(0, 0, 0)
                        self.msg_targetStatus = "?"
            # self.MeanPixelColor(croppedRgbImg)

# Nodes Publish
    def PublishEverything(self):
        self.pub_targetCroppedRgbImg.publish(self.msg_targetCroppedRgbImg)
        self.pub_targetCroppedDepthImg.publish(self.msg_targetCroppedDepthImg)
        self.pub_targetStatus.publish(self.msg_targetStatus)
        self.pub_targetPoint.publish(self.msg_targetPoint)
        self.pub_targetSkeletonImg.publish(self.msg_targetSkeletonImg)
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

            # Controls the number of samples of every body part for person registration
            if self.get_person_id == True:
                # Arbitrary number of samples
                nSamples = 1000
                if(self.operator.CheckNSamples(nSamples)):
                    self.get_person_id = False
                    self.operator.existsID = True

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