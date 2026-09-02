from glob import glob
from setuptools import find_packages, setup

package_name = "fast_lio_rebuild"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="z",
    maintainer_email="z@todo.todo",
    description="FAST-LIO bag rebuild tools with reference odometry sliding-window visualization.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "reference_window_node = fast_lio_rebuild.reference_window_node:main",
            "odom_bag_republisher_node = fast_lio_rebuild.odom_bag_republisher_node:main",
        ],
    },
)
