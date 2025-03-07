# mediapipe_track
- ROS2 package that applies [MediaPipe Pose solution](https://google.github.io/mediapipe/solutions/pose) 
- Important addition to MediaPipe Pose: ability to calculate and detect the person's 3D position and publish this information in ROS Topics so the robot can know its relative position from the person detected
- Synchronous processing in real-time frames can be enabled/disabled by a service
- Assynchronous processing responds to Action requests
- Tested in Ubuntu 22 with ROS Humble

## Installation

### Building

```bash
cd <ros2_ws>/src
git clone https://github.com/UtBotsAtHome-UTFPR/mediapipe_track.git
git checkout ros2-dev
cd ..
colcon build --symlink-install
```

### Dependencies
This package 

The code runs on Python 3.9 and depends on mediapipe 0.10.21 for the latest features. Install the requirements:

```bash
roscd mediapipe_track/src
pip3 install -r requirements.txt
```

**OBS**: because of permission problems with the model access in the library, mediapipe's libraries will not be located in a virtualenv yet

## Running
To run the Mediapipe pose estimation node:

```bash
ros2 run mediapipe_track mediapipe_node
```

Or using launchfiles:

```bash
ros2 launch mediapipe_track mediapipe_node.launch.py
```

With launchfiles you can specify the parameter values using any of the arguments in Command Line or other launchfiles (for instance, disabling *draw_skeleton_img* or *calculate_torso_point* could save processing usage):

```bash
'topic_namespace':
    Namespace for topics
    (default: '/utbots/vision')

'model_path':
    Path to the model file
    (default: '')

'num_poses':
    Maximum number of poses to detect
    (default: '1')

'detection_conf':
    Detection confidence threshold
    (default: '0.75')

'presence_conf':
    Presence confidence threshold
    (default: '0.5')

'track_conf':
    Tracking confidence threshold
    (default: '0.9')

'draw_skeleton_img':
    Draw skeleton image
    (default: 'true')

'calculate_torso_point':
    Calculate torso point
    (default: 'true')

'segmentation_mask':
    Use segmentation mask
    (default: 'false')

'rgb_topic':
    RGB input topic
    (default: '/camera/rgb/image_color')
```
