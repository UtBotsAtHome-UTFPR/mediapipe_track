import launch
import launch_ros.actions
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    return launch.LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument('topic_namespace', default_value='/utbots/vision', description='Namespace for topics'),
        DeclareLaunchArgument('model_path', default_value='', description='Path to the model file'),
        DeclareLaunchArgument('num_poses', default_value='1', description='Maximum number of poses to detect'),
        DeclareLaunchArgument('detection_conf', default_value='0.75', description='Detection confidence threshold'),
        DeclareLaunchArgument('presence_conf', default_value='0.5', description='Presence confidence threshold'),
        DeclareLaunchArgument('track_conf', default_value='0.9', description='Tracking confidence threshold'),
        DeclareLaunchArgument('draw_skeleton_img', default_value='true', description='Draw skeleton image'),
        DeclareLaunchArgument('calculate_torso_point', default_value='true', description='Calculate torso point'),
        DeclareLaunchArgument('segmentation_mask', default_value='false', description='Use segmentation mask'),
        DeclareLaunchArgument('rgb_topic', default_value='/camera/rgb/image_color', description='RGB input topic'),

        # Launch node with CLI-configurable parameters
        launch_ros.actions.Node(
            package='mediapipe_track',
            executable='mediapipe_node',
            name='mediapipe_node',
            output='screen',
            parameters=[{
                'topic_namespace': LaunchConfiguration('topic_namespace'),
                'model_path': LaunchConfiguration('model_path'),
                'num_poses': LaunchConfiguration('num_poses'),
                'detection_conf': LaunchConfiguration('detection_conf'),
                'presence_conf': LaunchConfiguration('presence_conf'),
                'track_conf': LaunchConfiguration('track_conf'),
                'draw_skeleton_img': LaunchConfiguration('draw_skeleton_img'),
                'calculate_torso_point': LaunchConfiguration('calculate_torso_point'),
                'segmentation_mask': LaunchConfiguration('segmentation_mask'),
                'rgb_topic': LaunchConfiguration('rgb_topic'),
            }]
        )
    ])
