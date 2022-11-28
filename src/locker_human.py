#!/usr/bin/env python
import cv2
from cv_bridge import CvBridge
import mediapipe as mp
import numpy as np
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, Point, TransformStamped
from sensor_msgs.msg import Image
from math import pow, sqrt, tan, radians


from tf.msg import tfMessage
from geometry_msgs.msg import TransformStamped
from tf.transformations import quaternion_from_euler

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

        # Calls main loop
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.75,
            min_tracking_confidence=0.9)
        self.mainLoop()

    def callback_rgbImg(self, msg):
        self.msg_rgbImg = msg
        self.newRgbImg = True
        # print("- RGB: new msg")

    def callback_depthImg(self, msg):
        self.msg_depthImg = msg
        self.newDepthImg = True
        # print("- Depth: new msg")

    def ProcessImg(self, msg_img):
        # Conversion to cv image
        cvImg = self.cvBridge.imgmsg_to_cv2(self.msg_rgbImg, "bgr8")

        # Not writeable passes by reference (more performance)
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

    def XyzToZxy(self, point):
        return Point(point.z, point.x, point.y)   

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

    def PublishEverything(self):
        self.pub_targetCroppedRgbImg.publish(self.msg_targetCroppedRgbImg)
        self.pub_targetCroppedDepthImg.publish(self.msg_targetCroppedDepthImg)
        self.pub_targetStatus.publish(self.msg_targetStatus)
        self.pub_targetPoint.publish(self.msg_targetPoint)
        self.pub_targetSkeletonImg.publish(self.msg_targetSkeletonImg)
        # self.SetupTfMsg(self.msg_targetPoint.point.x, self.msg_targetPoint.point.y, self.msg_targetPoint.point.z)

    def mainLoop(self):
        while rospy.is_shutdown() == False:
            self.loopRate.sleep()
            self.PublishEverything()
            print("\nTARGET")
            print(" - status: {}".format(self.msg_targetStatus))
            print(" - xyz: ({}, {}, {})".format(
                self.msg_targetPoint.point.x, 
                self.msg_targetPoint.point.y, 
                self.msg_targetPoint.point.z))
                
            # Else -> new RGB and new depth are true...
            if self.newRgbImg == True and self.newDepthImg == True:
                self.newRgbImg = False
                self.newDepthImg = False

                cv_rgbImg, poseResults = self.ProcessImg(self.msg_rgbImg)
                self.DrawLandmarks(cv_rgbImg, poseResults)
                self.msg_targetSkeletonImg = self.cvBridge.cv2_to_imgmsg(cv_rgbImg)
                # cv2.imshow('MediaPipe Pose', cv_rgbImg)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

                # If found landmarks...
                if poseResults.pose_landmarks:
                    torsoPoints = self.GetTorsoPoints(poseResults.pose_landmarks.landmark)
                    torsoCenter = self.GetPointsMean(torsoPoints)

                    cv_depthImg = self.cvBridge.imgmsg_to_cv2(self.msg_depthImg, "32FC1")
                    # cv2.imshow("depth Img", cv_depthImg)

                    try:
                        croppedRgbImg = self.CropTorsoImg(cv_rgbImg, "passthrough", torsoPoints, torsoCenter)
                        self.msg_targetCroppedRgbImg = self.cvBridge.cv2_to_imgmsg(croppedRgbImg)
                        # cv2.imshow("Cropped RGB", croppedRgbImg)
                    except:
                        print("------------- Error in RGB crop -------------")
                        continue
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
                        continue

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