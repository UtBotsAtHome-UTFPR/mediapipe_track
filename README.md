# mediapipe_track
This package applies MediaPipe Pose solution (https://google.github.io/mediapipe/solutions/pose) with Kinect V1 images through ROS Topics and Nodes. An important addition to MediaPipe Pose is the ability to calculate and detect the person's 3D position and publish this information in ROS Topics so the robot can know its relative position from the person detected.

## Installation

### Dependencies

This package depends on [freenect_launch](https://github.com/ros-drivers/freenect_stack) and runs on python, with mediapipe library.

**Building**
```
cd catkin_ws/src
git clone https://github.com/UtBot-UTFPR/mediapipe_track.git
cd ..
catkin_make
```
## Running

First, run freenect:

```
roslaunch freenect_launch freenect.launch
```

Then, to run the pose tracking and 3D position algorithm, run 

```
roslaunch mediapipe_track locker_human.launch
```
