# mediapipe_track
- ROS package that applies [MediaPipe Pose solution](https://google.github.io/mediapipe/solutions/pose) 
- Tested Kinect V1 RGB and Depth images 
- Important addition to MediaPipe Pose: ability to calculate and detect the person's 3D position and publish this information in ROS Topics so the robot can know its relative position from the person detected.

### Getting started
- #### Installation
    - **Dependencies**
        - This package depends on [freenect_launch](https://github.com/ros-drivers/freenect_stack) and runs on python, with mediapipe library.
    - **Building**
        ```
        cd catkin_ws/src
        git clone https://github.com/UtBotsAtHome-UTFPR/mediapipe_track.git
        cd ..
        catkin_make
        ```

    - **Pip requirements (skip if using Jetson + Ubuntu 18)**
        ```bash
        roscd mediapipe_track/src
        python3 -m pip install -r requirements.txt
        ```
    - **Only for Jetson Nano + Ubuntu 18**
        - Install Python 3.9 and virtualenv
            ```bash
            sudo add-apt-repository ppa:deadsnakes/ppa # Repository with many Python versions
            sudo apt update
            sudo apt install python3.9 python3.9-venv -y
            python3.9 -m pip install virtualenv
            PY_LOCATION=$(which python3.9)
            roscd mediapipe_track/src
            python3.9 -m virtualenv venv --python=$PY_LOCATION # Create virtual env
            source venv/bin/activate # Enter virtual env
            python -m pip install -r requirements.txt
            ```
        - You should **only use Mediapipe with the virtual Python executable**
            ```bash
            source venv/bin/activate # Enter virtual env
            python locker_human.py   # Instead of "rosrun mediapipe_track locker_human.py"
            ```

#### Running

- First, run freenect:
    ```
    roslaunch mediapipe_track freenect.launch
    ```
- Then, to run the Mediapipe pose estimation and 3d points positions:
    ```
    roslaunch mediapipe_track body_pose_and_points.launch
    ```
- To run only the Mediapipe pose estimation:
    ```
    rosrun mediapipe_track body_pose.py
    ```
- To view the 3D map with the published 3D point referred as the person detected position, run Rviz with:
    ```
    roslaunch mediapipe_track rviz.launch
    ```
