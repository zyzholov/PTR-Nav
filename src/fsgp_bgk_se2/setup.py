from glob import glob
from setuptools import find_packages, setup

package_name = "fsgp_bgk_se2"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="z",
    maintainer_email="z@example.com",
    description="FSGP-BGK and SEB traversability enhancement for SE2Grid terrain maps.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "fsgp_bgk_se2_node = fsgp_bgk_se2.fsgp_bgk_se2_node:main",
        ],
    },
)
