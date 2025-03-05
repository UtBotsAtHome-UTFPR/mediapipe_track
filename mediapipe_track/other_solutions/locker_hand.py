# https://google.github.io/mediapipe/solutions/hands.html
#!/usr/bin/env python
from os import stat
from re import X
import cv2
from cv_bridge import CvBridge, CvBridgeError
import mediapipe as mp
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from math import sqrt, pow, floor, ceil


class LockHand():
    def __init__(self, cam_id):
        # Camera
        self.videoCapture = cv2.VideoCapture(cam_id)

        # Messages
        self.msg_status = "None"
        self.msg_point = Point(0, 0, 0)
        self.msg_croppedDepthImg = None
        self.msg_depthImg = None

        # Publishers and Subscribers
        self.pub_status = rospy.Publisher(
            '/vision/locker_hand/status', String, queue_size=10)
        self.pub_point = rospy.Publisher(
            "/vision/locker_hand/point", Point, queue_size=10)
        self.pub_depthCroppedImg = rospy.Publisher(
            "/vision/locker_hand/img/depth/cropped", Image, queue_size=10)
        self.sub_depthImg = rospy.Subscriber(
            "/camera_rgb/depth/", String, self.callback_depthImg)

        # ROS node
        rospy.init_node('locker_hand', anonymous=True)

        # Time
        self.loopRate = rospy.Rate(30)
        self.t_lastHand = 0.0  # sec
        self.t_timeout = 0.250  # sec

        # Mediapipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        # Cv
        self.bridge = CvBridge()

        self.mainLoop()

    def callback_depthImg(self, msg):
        self.msg_depthImg = msg

    def GetHandPoint(self, landmarks):
        x = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x
        y = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
        z = 0
        return Point(x, y, z)

    def GetHandStatus(self, landmarks):
        if (landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y and landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].y > landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y):
            status = "Front"
        elif (landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y > landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y and landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].y < landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y):
            status = "Back"
        elif (landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x < landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x and landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].y < landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y):
            status = "Left"
        elif (landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x > landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x and landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].y < landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y):
            status = "Right"
        else:
            status = "Undefined"
        return status

    def GetHandDistanceToCenter(self, point):
        return sqrt(pow(point.x - 0.5, 2) + pow(point.y - 0.5, 2))

    def GetHandCenterOfGravity(self, landmarks):
        sum_x = 0
        sum_y = 0
        i = 0
        for landmark in landmarks.landmark:
            sum_x = sum_x + landmark.x
            sum_y = sum_y + landmark.y
            i = i + 1
        return Point(sum_x/i, sum_y/i, 0)


    def GetClosestPointToCenter(self, image, results, points, distancesToCenter, statuses, centers):
        smallestDistance = min(distancesToCenter)
        if smallestDistance != None:
            i = 0
            for distance in distancesToCenter:
                if distance == smallestDistance:
                    point = points[i]
                    status = statuses[i]
                    center = centers[i]
                    self.mp_drawing.draw_landmarks(
                        image, results.multi_hand_landmarks[i], self.mp_hands.HAND_CONNECTIONS)
                    print("\nCLOSEST HAND TO CENTER")
                    print(
                        " - Coordinates: ({}, {}, {})".format(point.x, point.y, point.z))
                    print(
                        " - Center: ({}, {}, {})".format(center.x, center.y, center.z))
                    print(" - status: {}".format(status))
                    self.msg_point = point
                    self.msg_status = status
                    self.Get3DPointFrom2D(center)
                    break
                i += 1

    def Get3DPointFrom2D(self, center):
        # tf_x = ?
        tf_y = self.msg_point.x
        tf_z = self.msg_point.y
        if self.msg_depthImg != None:
            cv_image = self.bridge.imgmsg_to_cv2(self.msg_depthImg, "32FC1")
            cropped_image = cv_image[floor(center.x-2):ceil(center.x+2), floor(center.y-2):ceil(center.y+2)]
            self.msg_croppedDepthImg = self.bridge.cv2_to_imgmsg(cropped_image, "passthrough")

            # cv_image = bridge.imgmsg_to_cv2(img, "32FC1")
            # print (cv_image[10,10])

    def PublishEverything(self):
        self.pub_status.publish(self.msg_status)
        self.pub_point.publish(self.msg_point)
        if self.msg_croppedDepthImg is not None:
            self.pub_depthCroppedImg.publish(self.msg_croppedDepthImg)

    def mainLoop(self):
        with self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while self.videoCapture.isOpened() and rospy.is_shutdown() == False:
                self.loopRate.sleep()

                # Checks if videocapture has frames
                success, image = self.videoCapture.read()
                if not success:
                    print("Empty camera frame")
                    continue

                # Flips horizontally and converts BGR to RGB
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

                # Not writeable passes by reference (more performance)
                image.flags.writeable = False

                # Image processing
                results = hands.process(image)

                # To draw the hand annotations on the image
                image.flags.writeable = True

                # Back to BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # If found landmarks...
                if results.multi_hand_landmarks:

                    # 2D point coordinates for all detected hands (each coordinate goes from 0 to 1)
                    points = []

                    # Distances from each detected hand to center of screen
                    distancesToCenter = []

                    # 'Front', 'Back', 'Left' or 'Right'
                    statuses = []

                    # Centers of mass
                    centers = []

                    for hand_landmarks in results.multi_hand_landmarks:
                        # self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        point = self.GetHandPoint(hand_landmarks)
                        points.append(point)
                        distancesToCenter.append(
                            self.GetHandDistanceToCenter(point))
                        statuses.append(self.GetHandStatus(hand_landmarks))
                        centers.append(self.GetHandCenterOfGravity(hand_landmarks))

                    self.GetClosestPointToCenter(
                        image, results, points, distancesToCenter, statuses, centers)

                # No hands detected
                else:
                    t_now = rospy.get_time()
                    if (t_now - self.t_lastHand > self.t_timeout and self.msg_status != "None"):
                        print("\nDefaulting to no hand")
                        self.t_lastHand = t_now
                        self.msg_point = Point(0, 0, 0)
                        self.msg_status = "None"

                cv2.imshow('hand control', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

                self.PublishEverything()
            self.videoCapture.release()


if __name__ == "__main__":
    lockHand = LockHand(0)
