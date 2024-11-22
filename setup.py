from setuptools import setup

package_name = 'mediapipe_track'

setup(
    name=package_name,
    version='0.0.0',
    packages=['mediapipe_track'],
    install_requires=['setuptools', 'rclpy'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'person_pose = mediapipe_track.person_pose:main',  # Entry point for the node
        ],
    },
)