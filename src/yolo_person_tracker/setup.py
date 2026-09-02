from glob import glob
from setuptools import find_packages, setup

package_name = "yolo_person_tracker"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="z",
    maintainer_email="z@todo.todo",
    description="YOLO based person detection, tracking, prediction, and annotated video publishing.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "person_tracker_node = yolo_person_tracker.person_tracker_node:main",
            "topdown_obstacle_node = yolo_person_tracker.topdown_obstacle_node:main",
        ],
    },
)
