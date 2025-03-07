import launch
import launch_ros.actions

def generate_launch_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='mediapipe_track',
            executable='mediapipe_node',
            name='mediapipe_node',
            output='screen',
            parameters=[
                {'topic_namespace': '/utbots/vision'},
                {'model_path': ''}, 
                {'num_poses': 1},
                {'detection_conf': 0.5},
                {'presence_conf': 0.5},
                {'track_conf': 0.5},
                {'draw_skeleton_img:': True},
                {'calculate_torso_point': True},
                {'segmentation_mask': False},
                {'rgb_topic': '/camera/rgb/image_color'}
            ]
        )
    ])