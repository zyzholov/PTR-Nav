#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="/home/z/workspace_zyz/seb_clean"
SRC_DIR="$ROOT_DIR/src"

# Stop stale Gazebo/spawn sessions to avoid mixed-state startup.
pkill -9 -x gzserver >/dev/null 2>&1 || true
pkill -9 -x gzclient >/dev/null 2>&1 || true
pkill -9 -f 'spawn_entity.py' >/dev/null 2>&1 || true
pkill -9 -f 'ros2 launch fake_ugv gazebo_vehicle_only.launch.py' >/dev/null 2>&1 || true
pkill -9 -f 'gazebo_vehicle_only.launch.py' >/dev/null 2>&1 || true
# Stop all stale ros2 CLI publishers; argument variants (-r/-1/--qos-*) are common
# and topic-specific patterns can miss them, causing startup self-driving.
pkill -9 -f 'ros2 topic pub' >/dev/null 2>&1 || true
# Stop stale helper controllers from previous sessions.
pkill -9 -f 'keyboard_ctrl.py' >/dev/null 2>&1 || true
pkill -9 -f 'startup_speed_hold.py' >/dev/null 2>&1 || true
pkill -9 -f 'cmd_vel_relay_node' >/dev/null 2>&1 || true
pkill -9 -f 'ros2 run fake_ugv keyboard_ctrl' >/dev/null 2>&1 || true
sleep 1

source /opt/ros/humble/setup.bash
if [[ -f "$ROOT_DIR/install/local_setup.bash" ]]; then
	source "$ROOT_DIR/install/local_setup.bash"
elif [[ -f "$ROOT_DIR/install/setup.bash" ]]; then
	set +e
	source "$ROOT_DIR/install/setup.bash"
	ROOT_SETUP_RC=$?
	set -e
	if [[ $ROOT_SETUP_RC -ne 0 ]]; then
		echo "[run_gazebo_vehicle] warning: failed to source $ROOT_DIR/install/setup.bash, trying local_setup.bash" >&2
		if [[ -f "$ROOT_DIR/install/local_setup.bash" ]]; then
			source "$ROOT_DIR/install/local_setup.bash" || true
		fi
	fi
fi

# Keep DDS middleware consistent with Gazebo-side ROS graph to avoid hidden topics.
if [[ -z "${RMW_IMPLEMENTATION:-}" ]]; then
	export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
fi

if [[ ! -d "$ROOT_DIR/install" ]]; then
	if [[ -f "$SRC_DIR/install/setup.bash" ]]; then
		source "$SRC_DIR/install/setup.bash"
	elif [[ -f "$SRC_DIR/install/local_setup.bash" ]]; then
		source "$SRC_DIR/install/local_setup.bash"
	fi
fi

if [[ -d "$ROOT_DIR/install" ]]; then
	export AMENT_PREFIX_PATH="$ROOT_DIR/install:${AMENT_PREFIX_PATH:-}"
	export CMAKE_PREFIX_PATH="$ROOT_DIR/install:${CMAKE_PREFIX_PATH:-}"
elif [[ -d "$SRC_DIR/install" ]]; then
	export AMENT_PREFIX_PATH="$SRC_DIR/install:${AMENT_PREFIX_PATH:-}"
	export CMAKE_PREFIX_PATH="$SRC_DIR/install:${CMAKE_PREFIX_PATH:-}"
fi

DEFAULT_URDF="$SRC_DIR/simulator/fake_ugv/urdf/car_ackermann.sdf"
LAUNCH_ARGS=("$@")
HAS_URDF_ARG=0
HAS_X_ARG=0
HAS_Y_ARG=0
HAS_Z_ARG=0
HAS_PAUSE_WORLD_ARG=0
HAS_ENABLE_STARTUP_HOLD_ARG=0
HAS_STARTUP_HOLD_SECONDS_ARG=0
WORLD_ARG=""

for arg in "${LAUNCH_ARGS[@]}"; do
	if [[ "$arg" == urdf:=* ]]; then
		HAS_URDF_ARG=1
	fi
	if [[ "$arg" == x:=* ]]; then
		HAS_X_ARG=1
	fi
	if [[ "$arg" == y:=* ]]; then
		HAS_Y_ARG=1
	fi
	if [[ "$arg" == z:=* ]]; then
		HAS_Z_ARG=1
	fi
	if [[ "$arg" == startup_hold_pause_world:=* ]]; then
		HAS_PAUSE_WORLD_ARG=1
	fi
	if [[ "$arg" == enable_startup_hold:=* ]]; then
		HAS_ENABLE_STARTUP_HOLD_ARG=1
	fi
	if [[ "$arg" == startup_hold_seconds:=* ]]; then
		HAS_STARTUP_HOLD_SECONDS_ARG=1
	fi
	if [[ "$arg" == world:=* ]]; then
		WORLD_ARG="${arg#world:=}"
	fi
done

if [[ $HAS_URDF_ARG -eq 0 ]]; then
	LAUNCH_ARGS=("urdf:=$DEFAULT_URDF" "${LAUNCH_ARGS[@]}")
fi

if [[ $HAS_PAUSE_WORLD_ARG -eq 0 ]]; then
	LAUNCH_ARGS+=("startup_hold_pause_world:=0")
fi

if [[ $HAS_STARTUP_HOLD_SECONDS_ARG -eq 0 ]]; then
	LAUNCH_ARGS+=("startup_hold_seconds:=0.0")
fi

if [[ $HAS_ENABLE_STARTUP_HOLD_ARG -eq 0 ]]; then
	LAUNCH_ARGS+=("enable_startup_hold:=true")
fi

if [[ $HAS_X_ARG -eq 0 ]]; then
	LAUNCH_ARGS+=("x:=0.0")
fi
if [[ $HAS_Y_ARG -eq 0 ]]; then
	LAUNCH_ARGS+=("y:=0.0")
fi
if [[ $HAS_Z_ARG -eq 0 ]]; then
	LAUNCH_ARGS+=("z:=0.0")
fi

SOURCE_LAUNCH_FILE="$SRC_DIR/simulator/fake_ugv/launch/gazebo_vehicle_only.launch.py"
if [[ -f "$SOURCE_LAUNCH_FILE" ]]; then
	exec ros2 launch "$SOURCE_LAUNCH_FILE" "${LAUNCH_ARGS[@]}"
fi

SRC_INSTALL_LAUNCH_FILE="$SRC_DIR/install/fake_ugv/share/fake_ugv/launch/gazebo_vehicle_only.launch.py"
if [[ -f "$SRC_INSTALL_LAUNCH_FILE" ]]; then
	exec ros2 launch "$SRC_INSTALL_LAUNCH_FILE" "${LAUNCH_ARGS[@]}"
fi

exec ros2 launch fake_ugv gazebo_vehicle_only.launch.py "${LAUNCH_ARGS[@]}"
