from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, ExecuteProcess, SetEnvironmentVariable
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
            os.path.join(fake_share, "maps", "rubicon"),
        ]
    except Exception:
        candidate_paths = []

    # Support source-tree maps when launching from a development workspace.
    launch_dir = os.path.dirname(os.path.abspath(__file__))
    source_pkg_root = os.path.abspath(os.path.join(launch_dir, ".."))
    candidate_paths.extend([
        os.path.join(source_pkg_root, "maps"),
        os.path.join(source_pkg_root, "maps", "mili_tech"),
        os.path.join(source_pkg_root, "maps", "rubicon"),
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
    fake_ugv_share = FindPackageShare("fake_ugv")
    gazebo_launch = PathJoinSubstitution([
        FindPackageShare("gazebo_ros"),
        "launch",
        "gazebo.launch.py",
    ])
    default_urdf_file = PathJoinSubstitution([fake_ugv_share, "urdf", "car_ackermann.sdf"])

    world = LaunchConfiguration("world")
    entity = LaunchConfiguration("entity")
    x = LaunchConfiguration("x")
    y = LaunchConfiguration("y")
    z = LaunchConfiguration("z")
    yaw = LaunchConfiguration("yaw")
    urdf_file = LaunchConfiguration("urdf")
    client_gui = LaunchConfiguration("client_gui")
    gazebo_master_uri = LaunchConfiguration("gazebo_master_uri")
    enable_startup_hold = LaunchConfiguration("enable_startup_hold")
    command_relay_rate = LaunchConfiguration("command_relay_rate")
    startup_hold_cmd_timeout = LaunchConfiguration("startup_hold_cmd_timeout")
    startup_hold_cmd_filter_alpha = LaunchConfiguration("startup_hold_cmd_filter_alpha")
    startup_hold_publish_max_linear_cmd = LaunchConfiguration("startup_hold_publish_max_linear_cmd")
    safe_ld = _safe_ld_library_path()
    startup_hold_enabled_default = "true"

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
            "GAZEBO_MASTER_URI": gazebo_master_uri,
            "GAZEBO_MODEL_DATABASE_URI": "",
            "GAZEBO_MODEL_PATH": model_path,
            "LIBGL_ALWAYS_SOFTWARE": os.environ.get("LIBGL_ALWAYS_SOFTWARE", "0"),
            "QT_X11_NO_MITSHM": "1",
            "LD_LIBRARY_PATH": safe_ld,
            "HOME": gzclient_home,
        },
        condition=IfCondition(client_gui),
    )

    spawn = ExecuteProcess(
        cmd=[
            "/usr/bin/python3",
            "/opt/ros/humble/lib/gazebo_ros/spawn_entity.py",
            "-entity", entity,
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
            "input_topic": "/car_0/cmd_vel",
            "output_topic": "/car_0/drive_cmd_vel",
            "publish_rate_hz": ParameterValue(command_relay_rate, value_type=float),
            "timeout_sec": ParameterValue(startup_hold_cmd_timeout, value_type=float),
            "filter_alpha": ParameterValue(startup_hold_cmd_filter_alpha, value_type=float),
            "max_linear_cmd": ParameterValue(startup_hold_publish_max_linear_cmd, value_type=float),
            "max_angular_cmd": 1.2,
        }],
        condition=IfCondition(enable_startup_hold),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "world",
            default_value=PathJoinSubstitution([fake_ugv_share, "worlds", "terrain_boxes.world"]),
            description="Gazebo world file",
        ),
        DeclareLaunchArgument("entity", default_value="car_0"),
        DeclareLaunchArgument("x", default_value="0.0"),
        DeclareLaunchArgument("y", default_value="0.0"),
        DeclareLaunchArgument("z", default_value="0.0"),
        DeclareLaunchArgument("yaw", default_value="0.0"),
        DeclareLaunchArgument("urdf", default_value=default_urdf_file),
        DeclareLaunchArgument("client_gui", default_value="true"),
        DeclareLaunchArgument("gazebo_master_uri", default_value="http://127.0.0.1:11345"),
        DeclareLaunchArgument("enable_startup_hold", default_value=startup_hold_enabled_default),
        DeclareLaunchArgument("command_relay_rate", default_value="50.0"),
        DeclareLaunchArgument("startup_hold_seconds", default_value="0.0"),
        DeclareLaunchArgument("startup_hold_kp", default_value="12.0"),
        DeclareLaunchArgument("startup_hold_max_cmd", default_value="2.5"),
        DeclareLaunchArgument("startup_hold_speed_deadband", default_value="0.008"),
        DeclareLaunchArgument("startup_hold_idle_brake_threshold", default_value="0.005"),
        DeclareLaunchArgument("startup_hold_idle_min_linear_cmd", default_value="0.20"),
        DeclareLaunchArgument("startup_hold_release_cmd_threshold", default_value="0.020"),
        DeclareLaunchArgument("startup_hold_cmd_timeout", default_value="0.30"),
        DeclareLaunchArgument("startup_hold_odom_timeout", default_value="0.35"),
        DeclareLaunchArgument("startup_hold_min_linear_cmd", default_value="0.0"),
        DeclareLaunchArgument("startup_hold_active_linear_kp", default_value="4.0"),
        DeclareLaunchArgument("startup_hold_active_max_linear_cmd", default_value="2.5"),
        DeclareLaunchArgument("startup_hold_idle_angular_kp", default_value="0.0"),
        DeclareLaunchArgument("startup_hold_idle_max_angular_cmd", default_value="0.0"),
        DeclareLaunchArgument("startup_hold_straight_angular_deadband", default_value="0.02"),
        DeclareLaunchArgument("startup_hold_yaw_damping_gain", default_value="0.0"),
        DeclareLaunchArgument("startup_hold_yaw_correction_max", default_value="0.0"),
        DeclareLaunchArgument("startup_hold_forward_cmd_scale_fwd", default_value="1.0"),
        DeclareLaunchArgument("startup_hold_forward_cmd_scale_rev", default_value="0.0"),
        DeclareLaunchArgument("startup_hold_reverse_cmd_scale_fwd", default_value="0.0"),
        DeclareLaunchArgument("startup_hold_reverse_cmd_scale_rev", default_value="1.0"),
        DeclareLaunchArgument("startup_hold_cmd_filter_alpha", default_value="0.35"),
        DeclareLaunchArgument("startup_hold_publish_max_linear_cmd", default_value="2.5"),
        DeclareLaunchArgument("startup_hold_pause_world", default_value="0"),
        SetEnvironmentVariable(name="GAZEBO_MASTER_URI", value=gazebo_master_uri),
        SetEnvironmentVariable(name="GAZEBO_MODEL_DATABASE_URI", value=""),
        SetEnvironmentVariable(name="GAZEBO_MODEL_PATH", value=model_path),
        LogInfo(msg=["Launching Gazebo vehicle-only mode (no planner/RViz changes). URDF: ", urdf_file]),
        LogInfo(msg=["Command relay: /car_0/cmd_vel -> /car_0/drive_cmd_vel at ", command_relay_rate, " Hz"]),
        gazebo,
        gzclient,
        spawn,
        command_relay,
    ])
