
import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, LogInfo, OpaqueFunction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _safe_ld_library_path():
    raw = os.environ.get("LD_LIBRARY_PATH", "")
    parts = []
    for item in raw.split(":"):
        path = item.strip()
        if not path:
            continue
        lower = path.lower()
        if "anaconda" in lower or "conda" in lower:
            continue
        if path not in parts:
            parts.append(path)

    required = [
        "/opt/ros/humble/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu/gazebo-11/plugins",
    ]
    for path in required:
        if path not in parts:
            parts.append(path)

    return ":".join(parts)


def _prepend_env(existing: str, value: str) -> str:
    items = [value]
    if existing:
        items.append(existing)
    return ":".join([item for item in items if item])


def _model_path_entries(*paths: str):
    seen = set()
    ordered = []
    for path in paths:
        if not path or not os.path.isdir(path) or path in seen:
            continue
        ordered.append(path)
        seen.add(path)
    return ordered


def _world_for_map(fake_ugv_share: str, map_name: str) -> str:
    if map_name == "mili" or map_name == "mili_tech":
        return os.path.join(fake_ugv_share, "maps", "mili_tech", "mili.world")
    if map_name == "rubicon":
        return os.path.join(fake_ugv_share, "maps", "rubicon", "rubicon.world")
    if map_name in ("mars_gale", "Mars_Gale", "gale"):
        return os.path.join(fake_ugv_share, "maps", "Mars_Gale", "mars_gale.world")
    if map_name == "terrain_boxes":
        return os.path.join(fake_ugv_share, "worlds", "terrain_boxes.world")
    return os.path.join(fake_ugv_share, "maps", "mili_tech", "mili.world")


def _resolve_world(fake_ugv_share: str, map_name: str, world_arg: str) -> str:
    world_arg = (world_arg or "").strip()
    if world_arg:
        return world_arg if os.path.isabs(world_arg) else os.path.abspath(world_arg)
    return _world_for_map(fake_ugv_share, (map_name or "").strip())


def _launch_gazebo(context, gazebo_share: str, hunter_share: str, fake_ugv_share: str):
    map_name = LaunchConfiguration("map").perform(context)
    world_arg = LaunchConfiguration("world").perform(context)
    world_path = _resolve_world(fake_ugv_share, map_name, world_arg)
    start_x = LaunchConfiguration("start_x").perform(context)
    start_y = LaunchConfiguration("start_y").perform(context)
    start_z = LaunchConfiguration("start_z").perform(context)
    is_named_rubicon = map_name == "rubicon" and not world_arg.strip()
    is_named_mars_gale = map_name in ("mars_gale", "Mars_Gale", "gale") and not world_arg.strip()
    if start_x == "auto":
        start_x = "7.5" if is_named_mars_gale else ("4.5" if is_named_rubicon else "1.8")
    if start_y == "auto":
        start_y = "-31.5" if is_named_mars_gale else ("-1.0" if is_named_rubicon else "0.0")
    if start_z == "auto":
        start_z = "0.12" if is_named_mars_gale else ("1.88" if is_named_rubicon else "0.30")
    terrain_lidar_profile = "mars_gale" if is_named_mars_gale else "normal"

    return [
        LogInfo(msg=f"Launching Hunter Gazebo world: {world_path}"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gazebo_share, "launch", "gazebo.launch.py")),
            launch_arguments={
                "world": world_path,
                "gui": "false",
                "verbose": "true",
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(hunter_share, "launch", "hunter_se_spawn.launch.py")),
            launch_arguments={
                "start_x": start_x,
                "start_y": start_y,
                "start_z": start_z,
                "start_yaw": LaunchConfiguration("start_yaw"),
                "robot_name": LaunchConfiguration("robot_name"),
                "drive_mode": LaunchConfiguration("drive_mode"),
                "terrain_lidar_profile": terrain_lidar_profile,
                "enable_d455": LaunchConfiguration("enable_d455"),
            }.items(),
        ),
    ]


def generate_launch_description():
    gazebo_share = get_package_share_directory("gazebo_ros")
    hunter_share = get_package_share_directory("hunter_se_gazebo")
    hunter_prefix = get_package_prefix("hunter_se_gazebo")
    livox_prefix = get_package_prefix("ros2_livox_simulation")
    fake_ugv_share = get_package_share_directory("fake_ugv")

    default_map = "mili"
    plugin_path = _prepend_env(
        os.environ.get("GAZEBO_PLUGIN_PATH", ""),
        ":".join([
            os.path.join(livox_prefix, "lib"),
            os.path.join(hunter_prefix, "lib"),
        ]),
    )
    model_entries = _model_path_entries(
        os.path.join(fake_ugv_share, "maps"),
        os.path.join(fake_ugv_share, "maps", "mili_tech"),
        "/usr/share/gazebo-11/models",
        "/usr/share/gazebo/models",
    )
    model_path = ":".join(model_entries)

    return LaunchDescription([
        DeclareLaunchArgument(
            "map",
            default_value=default_map,
            description="Named world preset: mili, rubicon, mars_gale, terrain_boxes. Used when world is empty.",
        ),
        DeclareLaunchArgument("world", default_value="", description="Explicit Gazebo world path. Overrides map when non-empty."),
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument(
            "software_render",
            default_value="false",
            description="Set true to force Mesa software rendering for gzclient.",
        ),
        DeclareLaunchArgument("gazebo_master_uri", default_value=os.environ.get("GAZEBO_MASTER_URI", "http://127.0.0.1:11345")),
        DeclareLaunchArgument("start_x", default_value="auto"),
        DeclareLaunchArgument("start_y", default_value="auto"),
        DeclareLaunchArgument("start_z", default_value="auto"),
        DeclareLaunchArgument("start_yaw", default_value="0.0"),
        DeclareLaunchArgument("robot_name", default_value="hunter_se"),
        DeclareLaunchArgument("drive_mode", default_value="hunter_joint"),
        DeclareLaunchArgument("enable_d455", default_value="true"),
        SetEnvironmentVariable(name="GAZEBO_MASTER_URI", value=LaunchConfiguration("gazebo_master_uri")),
        SetEnvironmentVariable(name="GAZEBO_MODEL_DATABASE_URI", value=""),
        SetEnvironmentVariable(name="GAZEBO_PLUGIN_PATH", value=plugin_path),
        SetEnvironmentVariable(name="GAZEBO_MODEL_PATH", value=model_path),
        LogInfo(msg="Starting Gazebo GUI client", condition=IfCondition(LaunchConfiguration("gui"))),
        ExecuteProcess(
            cmd=[
                "bash",
                "-lc",
                "rm -rf /tmp/hunter_se_gzclient_home && mkdir -p /tmp/hunter_se_gzclient_home && "
                "if [ \"${HUNTER_SE_SOFTWARE_RENDER:-false}\" = \"true\" ] || "
                "[ \"${HUNTER_SE_SOFTWARE_RENDER:-false}\" = \"1\" ]; then "
                "export LIBGL_ALWAYS_SOFTWARE=1; else unset LIBGL_ALWAYS_SOFTWARE; fi; "
                "gzclient --verbose",
            ],
            output="screen",
            additional_env={
                "GAZEBO_MASTER_URI": LaunchConfiguration("gazebo_master_uri"),
                "GAZEBO_MODEL_DATABASE_URI": "",
                "GAZEBO_MODEL_PATH": model_path,
                "GAZEBO_PLUGIN_PATH": plugin_path,
                "HUNTER_SE_SOFTWARE_RENDER": LaunchConfiguration("software_render"),
                "QT_X11_NO_MITSHM": "1",
                "LD_LIBRARY_PATH": _safe_ld_library_path(),
                "HOME": "/tmp/hunter_se_gzclient_home",
            },
            condition=IfCondition(LaunchConfiguration("gui")),
        ),
        OpaqueFunction(function=_launch_gazebo, args=[gazebo_share, hunter_share, fake_ugv_share]),
    ])
