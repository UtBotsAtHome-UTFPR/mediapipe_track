# mediapipe_track
- ROS package that applies [MediaPipe Pose solution](https://google.github.io/mediapipe/solutions/pose) 
- Tested Kinect V1 RGB and Depth images 
- Important addition to MediaPipe Pose: ability to calculate and detect the person's 3D position and publish this information in ROS Topics so the robot can know its relative position from the person detected.
- Support for use of person bounding box image

## Installation

### Building

```bash
cd catkin_ws/src
git clone https://github.com/UtBotsAtHome-UTFPR/mediapipe_track.git
cd ..
catkin_make
```

### Dependencies
This package depends on [freenect_launch](https://github.com/ros-drivers/freenect_stack) and runs on python, with mediapipe library.

The code runs on Python 3.8 and you must use a virtualenv (Install with `pip install virtualenv`) with the path `/usr/bin/mediapipe_track_env/bin/python` as the node expects its existence to run. Install the requirements:

```bash
cd /usr/bin
sudo python3 -m virtualenv mediapipe_track_env --python=$(which python3)
roscd mediapipe_track/src
/usr/bin/mediapipe_track_env/bin/python -m pip install -r requirements.txt
```

## Running

First, run freenect:

```bash
roslaunch mediapipe_track freenect.launch
```

Then, to run the Mediapipe pose estimation and 3d points positions:

```bash
roslaunch mediapipe_track body_pose_and_points.launch
```

To run only the Mediapipe pose estimation:

```bash
rosrun mediapipe_track body_pose.py
```

To view the 3D map with the published 3D point referred as the person detected position, run Rviz with:

```bash
roslaunch mediapipe_track rviz.launch
```
