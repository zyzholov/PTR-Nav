#!/usr/bin/env python3
"""Run the four modular launchers in an isolated ROS domain and retain logs."""

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", type=int, default=87)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--terrain-backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--dynamic-clear-enabled", choices=("true", "false"), default="true")
    parser.add_argument("--terrain-dynamic-clear-enabled", choices=("true", "false"), default="true")
    parser.add_argument("--disable-live-obstacles", action="store_true")
    parser.add_argument("--enable-map-expansion", action="store_true",
                        help="Experimental A/B mode; may erase adjacent static obstacles")
    parser.add_argument("--timeout", type=float, default=90.0, help="Per-stage wall-time limit")
    parser.add_argument("--skip-goal", action="store_true", help="Check data flow without moving the vehicle")
    parser.add_argument("--goal-lateral", type=float, default=0.0, help="Goal offset to the vehicle's left")
    parser.add_argument("--reverse-goal", action="store_true", help="Test a close goal behind the vehicle")
    parser.add_argument("--trial-scenario", choices=("none", "detour", "ramp", "step", "emerge", "crossing", "clearance", "clearance_guard"), default="none")
    parser.add_argument("--trial-timeout", type=float, default=45.0)
    parser.add_argument("--fault-stale-map", action="store_true", help="Pause terrain publishing during motion and verify planner stop")
    args = parser.parse_args()
    if (not 1 <= args.domain <= 100 or args.timeout <= 0 or args.trial_timeout <= 2
            or not math.isfinite(args.goal_lateral)):
        parser.error("use a free domain between 1 and 100, positive timeout, and finite goal-lateral")
    if args.fault_stale_map and (args.trial_scenario != "none" or args.skip_goal):
        parser.error("--fault-stale-map requires a goal and no trial scenario")

    env = os.environ.copy()
    env.update(ROS_DOMAIN_ID=str(args.domain), RMW_IMPLEMENTATION="rmw_fastrtps_cpp")
    log_dir = ROOT / "log" / ("modular_smoke_" + time.strftime("%Y%m%d_%H%M%S"))
    log_dir.mkdir(parents=True)
    processes = []
    stages = {}
    logs = []
    checks = []

    def start(name, command, ros=False):
        log_file = (log_dir / f"{name}.log").open("w")
        logs.append(log_file)
        if ros:
            command = ["bash", "-c", 'unset PYTHONHOME PYTHONPATH; source /opt/ros/humble/setup.bash && source install/setup.bash && exec "$@"', "probe", *command]
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log_file,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(process)
        return process

    def check(name, command, timeout):
        print(f"[smoke] {name}", flush=True)
        process = start(name, command, ros=True)
        code = process.wait(timeout=timeout)
        checks.append({"check": name, "exit_code": code})
        print((log_dir / f"{name}.log").read_text()[-5000:], end="", flush=True)
        if code:
            raise RuntimeError(f"{name} failed with exit code {code}")

    def wait_topic(topic):
        check("wait_" + topic.strip("/").replace("/", "_"),
              ["python3", "scripts/tools/hunter_stack_wait.py", topic, "--timeout", str(args.timeout)],
              args.timeout + 10)
        for name, process in stages.items():
            if process.poll() is not None:
                raise RuntimeError(f"{name} exited early with code {process.returncode}")

    result = 1
    try:
        print(f"[smoke] Domain={args.domain}; logs={log_dir}", flush=True)
        simulation_command = ["scripts/sim/start_simulation.sh", f"headless:={str(not args.gui).lower()}",
                              "--duration", "300", "--runtime-profile-period", "5"]
        if args.trial_scenario != "none" or args.fault_stale_map:
            simulation_command.extend(["--scene-profile", "light", "--map-asset", "none",
                                       "--nav-test-scenario", args.trial_scenario])
        stages["simulation"] = start("simulation", simulation_command)
        wait_topic("/livox/lidar")
        wait_topic("/imu/data_raw")
        stages["mapping"] = start("mapping", ["scripts/sim/start_mapping.sh", "rviz:=false",
                                             f"dynamic_expand_enabled:={str(args.enable_map_expansion).lower()}",
                                             f"dynamic_clear_enabled:={args.dynamic_clear_enabled}"])
        wait_topic("/Odometry")
        wait_topic("/Laser_map")
        parameter_command = ["python3", "scripts/sim/checks/mapping_parameters.py"]
        parameter_command.extend(["--dynamic-clear-enabled", args.dynamic_clear_enabled])
        if args.enable_map_expansion:
            parameter_command.append("--enable-map-expansion")
        check("mapping_parameters", parameter_command, 25)
        stages["terrain"] = start("terrain", ["scripts/sim/start_terrain_analysis.sh", f"terrain_backend:={args.terrain_backend}",
                                      f"terrain_dynamic_clear_enabled:={args.terrain_dynamic_clear_enabled}",
                                      "fsgp_gpu_backend:=" + ("cpu" if args.terrain_backend == "cpu" else "cupy"),
                                      f"rviz:={str(args.gui).lower()}",
                                      "grid_clouds:=true"])
        wait_topic("/car_0_terrain_analyzer/fused_map")
        check("wait_fsgp_collision_layer", ["python3", "scripts/tools/hunter_stack_wait.py",
                                             "/car_0_fsgp_bgk/fused_map", "--type", "grid",
                                             "--layer", "collision_risk", "--timeout", str(args.timeout)],
              args.timeout + 10)
        check("visualization_products", ["python3", "scripts/sim/checks/visualization_probe.py",
                                         "--timeout", "20"], 30)
        if args.gui:
            check("rviz_snapshot", ["python3", "scripts/sim/checks/rviz_snapshot.py",
                                     "--output", str(log_dir / "rviz_snapshot.png")], 20)
        stages["planning"] = start("planning", ["scripts/sim/start_planning_control.sh",
                                                f"live_obstacle_enable:={str(not args.disable_live_obstacles).lower()}"])
        if args.trial_scenario == "clearance_guard":
            check("map_guard_trial", ["python3", "scripts/sim/checks/map_guard_trial.py",
                                       "--timeout", str(args.trial_timeout)], args.trial_timeout + 20)
        elif not args.skip_goal:
            if args.fault_stale_map:
                check("stale_map", ["python3", "scripts/sim/checks/stale_map_trial.py",
                                    "--terrain-pgid", str(stages["terrain"].pid)], 50)
            elif args.trial_scenario == "none":
                goal_command = ["python3", "scripts/tools/hunter_goal_probe.py", "--distance", "2.0",
                                "--lateral", str(args.goal_lateral), "--timeout", "35",
                                "--check-visualization"]
                if args.gui:
                    goal_command.extend(["--rviz-snapshot", str(log_dir / "rviz_navigation.png")])
                if args.reverse_goal:
                    goal_command.append("--behind")
                check("goal", goal_command, 60)
            else:
                check("navigation_trial", ["python3", "scripts/sim/checks/navigation_trial.py",
                                           "--scenario", args.trial_scenario,
                                           "--distance", "9.0" if args.trial_scenario in ("emerge", "crossing", "clearance") else "7.0",
                                           "--timeout", str(args.trial_timeout)], args.trial_timeout + 35)
                if args.trial_scenario in ("emerge", "crossing", "clearance"):
                    planning_log = (log_dir / "planning.log").read_text()
                    simulation_log = (log_dir / "simulation.log").read_text()
                    event = ("navigation test obstacle has left the lane" if args.trial_scenario == "clearance"
                             else "navigation test obstacle has crossed the lane"
                             if args.trial_scenario == "crossing" else "navigation test obstacle is now blocking the lane")
                    if event not in simulation_log:
                        raise RuntimeError("moving obstacle did not enter the lane")
                    if args.trial_scenario != "clearance" and "emergency stop: new live obstacle" not in planning_log:
                        raise RuntimeError("live obstacle did not trigger an emergency replan")
        check("rates", ["scripts/sim/checks/rates.sh"], 60)
        check("map_rates", ["python3", "scripts/sim/checks/map_rates.py", "--min-rate",
                            "0.5" if args.terrain_backend == "cpu" else "2.0"], 20)
        check("drift", ["scripts/sim/checks/slam_drift.sh", "10"], 20)
        for name, process in stages.items():
            if process.poll() is not None:
                raise RuntimeError(f"{name} exited early with code {process.returncode}")
        result = 0
    except (RuntimeError, subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
        print(f"[smoke] FAILED: {exc}", file=sys.stderr, flush=True)
    finally:
        # Stop only process groups created by this run, including descendants
        # of ros2 launch. rclpy probes may consume SIGTERM, so bound shutdown.
        for process in reversed(processes):
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 8
        for process in reversed(processes):
            try:
                process.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        for process in reversed(processes):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        for log_file in logs:
            log_file.close()
        (log_dir / "result.json").write_text(json.dumps({"passed": result == 0, "scenario": args.trial_scenario,
                                                          "checks": checks}, indent=2) + "\n")
        print(f"[smoke] {'PASS' if result == 0 else 'FAIL'}; logs={log_dir}", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
