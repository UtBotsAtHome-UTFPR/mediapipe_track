import mediapipe as mp
import cv2
import numpy as np

class MediaPipePose():
    """
    A class that processes OpenCV format images using MediaPipe Pose estimation
    and predicts skeleton landmarks of detected persons in the image.

    ## Parameters:
    - `num_poses` (int)
    Maximum number of poses to detect. 
    - `detection_conf` (float)
    Confidence threshold for detecting a pose. 
    - `presence_conf` (float)
    Confidence threshold for presence estimation. 
    - `track_conf` (float)
    Confidence threshold for pose tracking. 
    - `segmentation_mask` (bool)
    Whether to include segmentation mask output. 
    """
    def __init__(self, num_poses=1, detection_conf=0.75, presence_conf=0.5, track_conf=0.9, segmentation_mask=False):
        self.num_poses = num_poses
        self.detection_conf = detection_conf
        self.presence_conf = presence_conf
        self.track_conf = track_conf
        self.segmentation_mask = segmentation_mask
        self.load_model(
            num_poses=self.num_poses,
            detection_conf=self.detection_conf,
            presence_conf=self.presence_conf,
            track_conf=self.track_conf,
            segmentation_mask=self.segmentation_mask
        )

    def load_model(self, num_poses=1, detection_conf=0.75, presence_conf=0.5, track_conf=0.9, segmentation_mask=False):
        """ Loads the MediaPipe Pose model with the selected parameters"""
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose

        # Initialize pose model
        self.pose = self.mp_pose.Pose(
            running_mode="IMAGE",
            num_poses=num_poses,
            min_detection_confidence=detection_conf,
            min_pose_presence_confidence=presence_conf,
            min_tracking_confidence=track_conf,
            output_segmentation_masks=segmentation_mask
        )

    def unload_model(self):
        """ Unloads the model and stops memory usage """
        if hasattr(self, 'pose'):
            self.pose.close()
            del self.pose

    def predict_landmarks(self, cv_image, draw=False):
        """ Performs landmark estimation in the image and draws them if requested """
        # Check if the image is in cv format
        if not isinstance(cv_image, (np.ndarray, np.generic)):
            raise ValueError("Input image must be a valid OpenCV image (numpy array).")

        # Verify if the model is loaded
        if not hasattr(self, 'pose'):
            self.load_model()
            raise RuntimeWarning("The pose model is not loaded. Loading now, in runtime.")

        # Process the results
        pose_results = self.pose.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        if draw:
            # To draw the hand annotations on the image
            cv_image.flags.writeable = True

            # Back to BGR
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

            # Draws the skeleton image
            self.mp_drawing.draw_landmarks(
            cv_image,
            pose_results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        else:
            # Make cv_image an empty image
            cv_image = np.zeros_like(cv_image)
        
        return pose_results.pose_landmarks, cv_image
        
    def calculate_representative_point(self, landmarks):
        """ Calculates the midpoint of the torso, with the shoulder and hips landmarks """
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]

        mean_x = (right_shoulder.x + left_shoulder.x + right_hip.x + left_hip.x) / 4
        mean_y = (right_shoulder.y + left_shoulder.y + right_hip.y + left_hip.y) / 4
        mean_z = (right_shoulder.z + left_shoulder.z + right_hip.z + left_hip.z) / 4

        return mean_x, mean_y, mean_z