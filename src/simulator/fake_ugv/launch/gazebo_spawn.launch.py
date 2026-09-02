from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def _safe_ld_library_path():
    raw = os.environ.get("LD_LIBRARY_PATH", "")
    parts = []
    for item in raw.split(":"):
        p = item.strip()
        if not p:
            continue
        lower = p.lower()
        if "anaconda" in lower or "conda" in lower:
            continue
        if p not in parts:
            parts.append(p)

    required = [
        "/opt/ros/humble/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu/gazebo-11/plugins",
    ]
    for p in required:
        if p not in parts:
            parts.append(p)

    return ":".join(parts)


def _safe_gazebo_model_path():
    raw = os.environ.get("GAZEBO_MODEL_PATH", "")
    parts = []
    for item in raw.split(":"):
        p = item.strip()
        if p and p not in parts:
            parts.append(p)

    try:
        fake_share = get_package_share_directory("fake_ugv")
        candidate_paths = [
            os.path.dirname(fake_share),
            fake_share,
            os.path.join(fake_share, "maps"),
            os.path.join(fake_share, "maps", "mili_tech"),
        ]
    except Exception:
        candidate_paths = []

    # Support source-tree maps when launching from a development workspace.
    launch_dir = os.path.dirname(os.path.abspath(__file__))
    source_pkg_root = os.path.abspath(os.path.join(launch_dir, ".."))
    candidate_paths.extend([
        os.path.join(source_pkg_root, "maps"),
        os.path.join(source_pkg_root, "maps", "mili_tech"),
    ])

    defaults = [
        "/usr/share/gazebo-11/models",
        "/usr/share/gazebo/models",
    ]

    for p in candidate_paths + defaults:
        if p and os.path.isdir(p) and p not in parts:
            parts.append(p)

    return ":".join(parts)


def generate_launch_description():
    world = LaunchConfiguration("world")
    x = LaunchConfiguration("x")
    y = LaunchConfiguration("y")
    z = LaunchConfiguration("z")
    yaw = LaunchConfiguration("yaw")
    urdf_file = LaunchConfiguration("urdf")
    client_gui = LaunchConfiguration("client_gui")
    gazebo_master_uri = LaunchConfiguration("gazebo_master_uri")
    enable_cmd_relay = LaunchConfiguration("enable_cmd_relay")
    cmd_topic = LaunchConfiguration("cmd_topic")
    drive_topic = LaunchConfiguration("drive_topic")
    command_relay_rate = LaunchConfiguration("command_relay_rate")
    command_relay_timeout = LaunchConfiguration("command_relay_timeout")
    command_filter_alpha = LaunchConfiguration("command_filter_alpha")

    fake_ugv_share = FindPackageShare("fake_ugv")
    gazebo_launch = PathJoinSubstitution([FindPackageShare("gazebo_ros"), "launch", "gazebo.launch.py"])
    default_urdf_file = PathJoinSubstitution([fake_ugv_share, "urdf", "car_ackermann.sdf"])
    safe_ld = _safe_ld_library_path()

    # Use local model set and disable online model database lookup to avoid GUI startup stalls.
    model_path = _safe_gazebo_model_path()
    gzclient_home = "/tmp/fake_ugv_gzclient_home"

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch),
        launch_arguments={
            "world": world,
            # Start server only; launch GUI client separately in safe mode.
            "gui": "false",
        }.items(),
    )

    gzclient = ExecuteProcess(
        cmd=["bash", "-lc", f"rm -rf {gzclient_home} && mkdir -p {gzclient_home} && gzclient"],
        output="screen",
        additional_env={
            # These options improve stability on mixed Conda/OpenGL environments.
            "GAZEBO_MASTER_URI": gazebo_master_uri,
            "GAZEBO_MODEL_DATABASE_URI": "",
            "GAZEBO_MODEL_PATH": model_path,
            "LIBGL_ALWAYS_SOFTWARE": os.environ.get("LIBGL_ALWAYS_SOFTWARE", "0"),
            "QT_X11_NO_MITSHM": "1",
            "LD_LIBRARY_PATH": safe_ld,
            # Isolate from stale ~/.gazebo GUI states that can lock camera or freeze startup.
            "HOME": gzclient_home,
        },
        condition=IfCondition(client_gui),
    )

    spawn = ExecuteProcess(
        cmd=[
            "/usr/bin/python3",
            "/opt/ros/humble/lib/gazebo_ros/spawn_entity.py",
            "-entity", "car_0",
            "-file", urdf_file,
            "-x", x,
            "-y", y,
            "-z", z,
            "-Y", yaw,
        ],
        output="screen",
    )

    command_relay = Node(
        package="fake_ugv",
        executable="cmd_vel_relay_node",
        name="car_0_cmd_vel_relay",
        output="screen",
        parameters=[{
            "input_topic": cmd_topic,
            "output_topic": drive_topic,
            "publish_rate_hz": ParameterValue(command_relay_rate, value_type=float),
            "timeout_sec": ParameterValue(command_relay_timeout, value_type=float),
            "filter_alpha": ParameterValue(command_filter_alpha, value_type=float),
            "max_linear_cmd": 2.8,
            "max_angular_cmd": 1.2,
        }],
        condition=IfCondition(enable_cmd_relay),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "world",
            default_value=PathJoinSubstitution([fake_ugv_share, "worlds", "terrain_boxes.world"]),
            description="Gazebo world file",
        ),
        DeclareLaunchArgument("x", default_value="1.8"),
        DeclareLaunchArgument("y", default_value="0.0"),
        DeclareLaunchArgument("z", default_value="0.0"),
        DeclareLaunchArgument("yaw", default_value="0.0"),
        DeclareLaunchArgument("urdf", default_value=default_urdf_file),
        DeclareLaunchArgument("client_gui", default_value="true"),
        DeclareLaunchArgument("gazebo_master_uri", default_value="http://127.0.0.1:11345"),
        DeclareLaunchArgument("enable_cmd_relay", default_value="true"),
        DeclareLaunchArgument("cmd_topic", default_value="/car_0/cmd_vel"),
        DeclareLaunchArgument("drive_topic", default_value="/car_0/drive_cmd_vel"),
        DeclareLaunchArgument("command_relay_rate", default_value="50.0"),
        DeclareLaunchArgument("command_relay_timeout", default_value="0.30"),
        DeclareLaunchArgument("command_filter_alpha", default_value="0.35"),
        SetEnvironmentVariable(name="GAZEBO_MASTER_URI", value=gazebo_master_uri),
        SetEnvironmentVariable(name="GAZEBO_MODEL_DATABASE_URI", value=""),
        SetEnvironmentVariable(name="GAZEBO_MODEL_PATH", value=model_path),
        gazebo,
        gzclient,
        spawn,
        command_relay,
    ])
