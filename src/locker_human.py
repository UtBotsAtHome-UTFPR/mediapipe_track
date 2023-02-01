#!/usr/bin/env python

import os

# Mediapipe imports
import cv2
import mediapipe as mp

# Image processing
from cv_bridge import CvBridge
from sklearn.cluster import KMeans
from collections import Counter

# ROS
import rospy
## Message definitions
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, Point, TransformStamped
from sensor_msgs.msg import Image
## Transformation tree
from tf.msg import tfMessage
from geometry_msgs.msg import TransformStamped
from tf.transformations import quaternion_from_euler

# Math 
import numpy as np
from math import pow, sqrt, tan, radians

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

        # To tell if there's a new msg
        self.newRgbImg = False
        self.newDepthImg = False

        # Publishers and Subscribers
        self.pub_tf = rospy.Publisher(
            "/tf", tfMessage, queue_size=1)
        self.pub_targetStatus = rospy.Publisher(
            "/apollo/vision/lock/target/status", String, queue_size=10)
        self.pub_targetPoint = rospy.Publisher(
            "/apollo/vision/lock/target/point", PointStamped, queue_size=10)
        self.pub_targetCroppedRgbImg = rospy.Publisher(
            "/apollo/vision/lock/target/rgb", Image, queue_size=10)
        self.pub_targetCroppedDepthImg = rospy.Publisher(
            "/apollo/vision/lock/target/depth", Image, queue_size=10)
        self.pub_targetSkeletonImg = rospy.Publisher(
            "/apollo/vision/lock/target/skeletonImage", Image, queue_size=10)
        self.sub_rgbImg = rospy.Subscriber(
            topic_rgbImg, Image, self.callback_rgbImg)
        self.sub_depthImg = rospy.Subscriber(
            topic_depthImg, Image, self.callback_depthImg)

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
        self.MIN_VISIBILITY = 0.5

        # Statistics
        ## 0 - Shoulder Width
        ## 1 - Hip Width
        ## 2 - Torso Height
        ## 3 - Right Leg
        ## 4 - Left Leg
        ## 5 - Torso Color
        
        self.iteration = 0
        self.meanList = [0,0,0,0,0,0]
        self.minList = [0,0,0,0,0,0]
        self.maxList = [0,0,0,0,0,0]
        self.varianceList = [0,0,0,0,0,0]

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
    def LimbsDistances(self, landmark):

        rHip = landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        lHip = landmark[self.mp_pose.PoseLandmark.LEFT_HIP] 
        rShoulder = landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lShoulder = landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        rKnee = landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        lKnee = landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
        rAnkle = landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        lAnkle = landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]

        os.system("clear")
        # print(
        #     "Valued points:\n- Right Hip: {}\n- Left Hip: {}\n- Right Shoulder: {}\n- Left Shoulder: {}".format(
        #         rHip, 
        #         lHip, 
        #         rShoulder,
        #         lShoulder))

        # print("Shoulder:")
        shoulderWidth = self.EucDist(landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER], landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER])
        print("SW: {:.2f}".format(shoulderWidth))
        # print("\nHip:")
        hipWidth = self.EucDist(landmark[self.mp_pose.PoseLandmark.RIGHT_HIP], landmark[self.mp_pose.PoseLandmark.LEFT_HIP])
        print("HW: {:.2f}".format(hipWidth))    
        
        # print("\nTorso:")
        if(self.ChkVisib(rHip) and self.ChkVisib(lHip) and self.ChkVisib(lShoulder) and self.ChkVisib(lShoulder)):
            torsoHeightR = self.EucDist(rHip, rShoulder)
            # print("RTH: {:.2f}\n".format(torsoHeightR))   
            torsoHeightL = self.EucDist(lHip, lShoulder)
            # print("LTH: {:.2f}\n".format(torsoHeightL))  
            meanTorsoHeight = (torsoHeightL+torsoHeightR)/2
            print("Mean TH: {:.2f}".format(meanTorsoHeight))
            # self.Statistics(landmark, 0, meanTorsoHeight)
        # else:
        #     print("WARN: Points not visible")

        # print("\nRight Leg:")
        if(self.ChkVisib(rHip) and self.ChkVisib(rKnee) and self.ChkVisib(rAnkle)):
            rLegLenght = self.EucDist(rHip, rKnee)+self.EucDist(rKnee,rAnkle)
            print("Right Leg Lenght: {:.2f}".format(rLegLenght))
            # self.Statistics(landmark, 0, rLegLenght)
        # else:
        #     print("WARN: Points not visible")

        # print("\nLeft Leg:")
        if(self.ChkVisib(lHip) and self.ChkVisib(lKnee) and self.ChkVisib(lAnkle)):
            lLegLenght = self.EucDist(lHip, lKnee)+self.EucDist(lKnee,lAnkle)
            print("Left Leg Lenght: {:.2f}".format(lLegLenght))
            # self.Statistics(landmark, 0, lLegLenght)
        # else:
        #     print("WARN: Points not visible")

        

## Torso
    ''' Gets points for torso (shoulders and hips) '''
    def GetTorsoPoints(self, landmark):
        rightShoulder = Point(
            landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x, 
            landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y, 
            landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].z)
        # print(" - RSHOULDER xyz: ({}, {}, {})".format(
        #     landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x, 
        #     landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y, 
        #     landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].z))
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

    def Statistics(self, landmark, feature, value):
        if self.iteration < 1:
            firstValue = (self.EucDist(landmark[self.mp_pose.PoseLandmark.RIGHT_HIP], landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER])+self.EucDist(landmark[self.mp_pose.PoseLandmark.LEFT_HIP], landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]))/2
            self.meanList[feature] = firstValue
            self.maxList[feature] = firstValue
            self.minList[feature] = firstValue
            self.varianceList[feature] = firstValue
            self.iteration=1
        else:
            # Moving Average of Streaming Data
            self.meanList[feature] = self.meanList[feature]+(value-self.meanList[feature])/self.iteration
            ##relevance = 1/self.iteration
            ##self.meanList[feature] = (self.meanList[feature]+value*relevance)/(1+relevance)

            # Maximum Value
            if value > self.maxList[feature]:
                self.maxList[feature] = value

            # Maximum Value
            if value < self.minList[feature]:
                self.minList[feature] = value
            
            # Variance
            self.varianceList[feature] = self.maxList[feature] - self.minList[feature]

            # Iteration
            self.iteration+=1

            print("Iteration: {}\nGMean: {}\nGMin: {}\nGMax: {}\nGVar: {}".format(self.iteration, 
            self.meanList[feature],self.minList[feature],self.maxList[feature],self.varianceList[feature]))
            
    def ChkVisib(self, lmark):
        return (lmark.visibility > self.MIN_VISIBILITY)

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

# Image processing
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

        return list(dominant_color)
        
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

            # Basic RGB only TEST
            if self.newRgbImg == True:
                cv_rgbImg, poseResults = self.ProcessImg(self.msg_rgbImg)
                self.DrawLandmarks(cv_rgbImg, poseResults)
                self.msg_targetSkeletonImg = self.cvBridge.cv2_to_imgmsg(cv_rgbImg)
                # cv2.imshow('MediaPipe Pose', cv_rgbImg)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
    
                # if poseResults.pose_landmarks:
                #     self.LimbsDistances(poseResults.pose_world_landmarks.landmark)

                if poseResults.pose_landmarks:
                    torsoPoints = self.GetTorsoPoints(poseResults.pose_landmarks.landmark)
                    torsoCenter = self.GetPointsMean(torsoPoints)

                    try:
                        croppedRgbImg = self.CropTorsoImg(cv_rgbImg, "passthrough", torsoPoints, torsoCenter)
                        self.msg_targetCroppedRgbImg = self.cvBridge.cv2_to_imgmsg(croppedRgbImg)
                        cv2.imshow("Cropped RGB", croppedRgbImg)
                    except:
                        print("------------- Error in RGB crop -------------")
                        continue

                    # self.MeanPixelColor(croppedRgbImg)

                    height=512
                    width=512
                    blank_image = np.zeros((height,width,3), np.uint8)
                    blank_image[:]=self.get_dominant_color(croppedRgbImg)
                    cv2.imshow('3 Channel Window', blank_image)
                
            # Else -> new RGB and new depth are true...
            # if self.newRgbImg == True and self.newDepthImg == True:
            #     self.newRgbImg = False
            #     self.newDepthImg = False

            #     cv_rgbImg, poseResults = self.ProcessImg(self.msg_rgbImg)
            #     self.DrawLandmarks(cv_rgbImg, poseResults)
            #     self.msg_targetSkeletonImg = self.cvBridge.cv2_to_imgmsg(cv_rgbImg)
            #     # cv2.imshow('MediaPipe Pose', cv_rgbImg)
            #     if cv2.waitKey(5) & 0xFF == 27:
            #         break

            #     # If found landmarks...
            #     if poseResults.pose_world_landmarks:
            #         torsoPoints = self.GetTorsoPoints(poseResults.pose_landmarks.landmark)
            #         torsoCenter = self.GetPointsMean(torsoPoints)

            #         cv_depthImg = self.cvBridge.imgmsg_to_cv2(self.msg_depthImg, "32FC1")
            #         # cv2.imshow("depth Img", cv_depthImg)

            #         try:
            #             croppedRgbImg = self.CropTorsoImg(cv_rgbImg, "passthrough", torsoPoints, torsoCenter)
            #             self.msg_targetCroppedRgbImg = self.cvBridge.cv2_to_imgmsg(croppedRgbImg)
            #             # cv2.imshow("Cropped RGB", croppedRgbImg)
            #         except:
            #             print("------------- Error in RGB crop -------------")
            #             continue
            #         try:
            #             croppedDepthImg = self.CropTorsoImg(cv_depthImg, "32FC1", torsoPoints, torsoCenter)
            #             self.msg_targetCroppedDepthImg = self.cvBridge.cv2_to_imgmsg(croppedDepthImg)
            #             # cv2.imshow("Cropped Depth", croppedDepthImg)
            #             torsoCenter3d = self.Get3dPointFromDepthPixel(torsoCenter, self.GetTorsoDistance(croppedDepthImg))
            #             torsoCenter3d = self.XyzToZxy(torsoCenter3d)
            #             # self.msg_targetPoint = Point(self.GetTorsoDistance(croppedDepthImg), 0, 0)
            #             self.msg_targetPoint.point = torsoCenter3d
            #             self.msg_targetStatus = "Located"
            #         except:
            #             print("------------- Error in depth crop -------------")
            #             continue

                # Nothing detected...
                else:
                    t_now = rospy.get_time()
                    if (t_now - self.t_last > self.t_timeout and self.msg_targetStatus != "?"):
                        self.t_last = t_now
                        self.msg_targetPoint.point = Point(0, 0, 0)
                        self.msg_targetStatus = "?"

if __name__ == "__main__":
    lockHand = LockPose(
        "/camera/rgb/image_raw",
        "/camera/depth_registered/image_raw",
        43,
        57)