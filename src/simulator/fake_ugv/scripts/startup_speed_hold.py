#!/usr/bin/python3

import argparse
import math
import subprocess

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


class StartupSpeedHold(Node):
    def __init__(
        self,
        cmd_topic: str,
        drive_topic: str,
        drive_topic_reverse: str,
        odom_topic: str,
        rate_hz: float,
        hold_seconds: float,
        kp: float,
        max_cmd: float,
        speed_deadband: float,
        idle_brake_threshold: float,
        idle_min_linear_cmd: float,
        release_cmd_threshold: float,
        cmd_timeout: float,
        odom_timeout: float,
        min_linear_cmd: float,
        active_linear_kp: float,
        active_max_linear_cmd: float,
        idle_angular_kp: float,
        idle_max_angular_cmd: float,
        straight_angular_deadband: float,
        yaw_damping_gain: float,
        yaw_correction_max: float,
        forward_cmd_scale_fwd: float,
        forward_cmd_scale_rev: float,
        reverse_cmd_scale_fwd: float,
        reverse_cmd_scale_rev: float,
        cmd_filter_alpha: float,
        pause_world_during_hold: bool,
        publish_max_linear_cmd: float,
    ) -> None:
        super().__init__("startup_speed_hold")
        self.cmd_topic = cmd_topic
        self.drive_topic = drive_topic
        self.drive_topic_reverse = drive_topic_reverse.strip()
        self.pub = self.create_publisher(Twist, drive_topic, 20)
        self.pub_reverse = None
        if self.drive_topic_reverse:
            self.pub_reverse = self.create_publisher(Twist, self.drive_topic_reverse, 20)
        self.sub = self.create_subscription(Twist, cmd_topic, self._on_cmd_msg, 20)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self._on_odom, 20)
        self.timer = self.create_timer(1.0 / rate_hz, self._on_timer)

        self.hold_seconds = max(0.0, float(hold_seconds))
        self.release_cmd_threshold = max(0.0, float(release_cmd_threshold))
        self.cmd_timeout = max(0.05, float(cmd_timeout))
        self.odom_timeout = max(0.05, float(odom_timeout))
        self.min_linear_cmd = max(0.0, float(min_linear_cmd))
        self.active_linear_kp = max(0.0, float(active_linear_kp))
        self.active_max_linear_cmd = max(0.0, float(active_max_linear_cmd))
        self.idle_linear_kp = float(kp)
        self.idle_max_linear_cmd = max(0.0, float(max_cmd))
        self.idle_angular_kp = max(0.0, float(idle_angular_kp))
        self.idle_max_angular_cmd = max(0.0, float(idle_max_angular_cmd))
        self.speed_deadband = max(0.0, float(speed_deadband))
        self.idle_brake_threshold = max(self.speed_deadband, float(idle_brake_threshold))
        self.idle_brake_release_threshold = max(self.speed_deadband * 0.8, self.idle_brake_threshold * 0.45)
        self.idle_min_linear_cmd = max(0.0, float(idle_min_linear_cmd))
        self.idle_position_kp = max(0.0, self.idle_linear_kp * 0.8)
        self.idle_position_deadband = max(0.004, self.idle_brake_threshold * 0.6)
        self.idle_vx_alpha = 0.2
        self.straight_angular_deadband = max(0.0, float(straight_angular_deadband))
        self.yaw_damping_gain = max(0.0, float(yaw_damping_gain))
        self.yaw_correction_max = max(0.0, float(yaw_correction_max))
        self.forward_cmd_scale_fwd = max(0.0, float(forward_cmd_scale_fwd))
        self.forward_cmd_scale_rev = max(0.0, float(forward_cmd_scale_rev))
        self.reverse_cmd_scale_fwd = max(0.0, float(reverse_cmd_scale_fwd))
        self.reverse_cmd_scale_rev = max(0.0, float(reverse_cmd_scale_rev))
        self.cmd_filter_alpha = self._clamp(float(cmd_filter_alpha), 0.0, 1.0)
        self.publish_max_linear_cmd = max(0.05, float(publish_max_linear_cmd))
        self.same_topic_mode = (self.cmd_topic == self.drive_topic)
        self.same_topic_echo_window = 0.12
        self.same_topic_match_eps = 1e-3
        self.last_guard_pub_time = None
        self.last_guard_linear = 0.0
        self.last_guard_angular = 0.0
        self.filtered_fwd_linear = 0.0
        self.filtered_rev_linear = 0.0
        self.filtered_fwd_angular = 0.0
        self.filtered_rev_angular = 0.0

        self.pause_world_during_hold = bool(pause_world_during_hold)
        self.world_paused_by_me = False

        self.start_time = self.get_clock().now()
        self.last_cmd = Twist()
        self.last_cmd_time = None
        self.have_odom = False
        self.last_odom_time = None
        self.vx = 0.0
        self.wz = 0.0
        self.px = 0.0
        self.py = 0.0
        self.yaw = 0.0
        self.have_pose = False
        self.idle_anchor_valid = False
        self.idle_anchor_x = 0.0
        self.idle_anchor_y = 0.0
        self.idle_anchor_yaw = 0.0

        self.get_logger().info(
            f"Startup speed hold enabled for {self.hold_seconds:.1f}s, output topic={drive_topic}"
        )
        if self.pub_reverse is not None:
            self.get_logger().info(
                f"Reverse routing enabled, reverse output topic={self.drive_topic_reverse}"
            )
        self.get_logger().info(f"Accepted command topic: {cmd_topic}")
        if self.pause_world_during_hold and self.hold_seconds > 0.0:
            if self._set_world_pause(True):
                self.world_paused_by_me = True
                self.get_logger().info("Gazebo world paused during startup hold")
            else:
                self.get_logger().warn(
                    "Failed to pause Gazebo world, fallback to zero-command hold only"
                )

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _set_idle_anchor(self) -> None:
        if not self.have_pose:
            self.idle_anchor_valid = False
            return
        self.idle_anchor_x = self.px
        self.idle_anchor_y = self.py
        self.idle_anchor_yaw = self.yaw
        self.idle_anchor_valid = True

    def _publish_cmd(self, linear_x: float, angular_z: float) -> None:
        publish_linear = float(linear_x)
        publish_angular = float(angular_z)

        self.last_guard_pub_time = self.get_clock().now()
        self.last_guard_linear = publish_linear
        self.last_guard_angular = publish_angular

        if self.pub_reverse is None:
            publish_linear = self._clamp(
                publish_linear,
                -self.publish_max_linear_cmd,
                self.publish_max_linear_cmd,
            )
            if self.cmd_filter_alpha > 0.0:
                self.filtered_fwd_linear += self.cmd_filter_alpha * (publish_linear - self.filtered_fwd_linear)
                self.filtered_fwd_angular += self.cmd_filter_alpha * (publish_angular - self.filtered_fwd_angular)
                out_linear = self.filtered_fwd_linear
                out_angular = self.filtered_fwd_angular
            else:
                out_linear = publish_linear
                out_angular = publish_angular

            msg = Twist()
            msg.linear.x = out_linear
            msg.angular.z = out_angular
            self.pub.publish(msg)
            return

        target_fwd_linear = 0.0
        target_rev_linear = 0.0
        msg_fwd = Twist()
        msg_rev = Twist()
        target_fwd_angular = publish_angular
        target_rev_angular = publish_angular

        if publish_linear > self.release_cmd_threshold:
            target_fwd_linear = publish_linear * self.forward_cmd_scale_fwd
            target_rev_linear = publish_linear * self.forward_cmd_scale_rev
        elif publish_linear < -self.release_cmd_threshold:
            target_fwd_linear = publish_linear * self.reverse_cmd_scale_fwd
            target_rev_linear = publish_linear * self.reverse_cmd_scale_rev

        target_fwd_linear = self._clamp(
            target_fwd_linear,
            -self.publish_max_linear_cmd,
            self.publish_max_linear_cmd,
        )
        target_rev_linear = self._clamp(
            target_rev_linear,
            -self.publish_max_linear_cmd,
            self.publish_max_linear_cmd,
        )

        if self.cmd_filter_alpha > 0.0:
            self.filtered_fwd_linear += self.cmd_filter_alpha * (target_fwd_linear - self.filtered_fwd_linear)
            self.filtered_rev_linear += self.cmd_filter_alpha * (target_rev_linear - self.filtered_rev_linear)
            self.filtered_fwd_angular += self.cmd_filter_alpha * (target_fwd_angular - self.filtered_fwd_angular)
            self.filtered_rev_angular += self.cmd_filter_alpha * (target_rev_angular - self.filtered_rev_angular)

            msg_fwd.linear.x = self.filtered_fwd_linear
            msg_rev.linear.x = self.filtered_rev_linear
            msg_fwd.angular.z = self.filtered_fwd_angular
            msg_rev.angular.z = self.filtered_rev_angular
        else:
            msg_fwd.linear.x = target_fwd_linear
            msg_rev.linear.x = target_rev_linear
            msg_fwd.angular.z = target_fwd_angular
            msg_rev.angular.z = target_rev_angular

        self.pub.publish(msg_fwd)
        self.pub_reverse.publish(msg_rev)

    def _set_world_pause(self, pause: bool) -> bool:
        msg = "pause: true" if pause else "pause: false"
        try:
            proc = subprocess.run(
                ["gz", "topic", "-p", "/gazebo/default/world_control", "-m", msg],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
                check=False,
            )
            return proc.returncode == 0
        except Exception:
            return False

    def _resume_world_if_paused(self, reason: str) -> None:
        if not self.world_paused_by_me:
            return
        if self._set_world_pause(False):
            self.get_logger().info(reason)
        else:
            self.get_logger().warn("Failed to resume Gazebo world, please unpause manually")
        self.world_paused_by_me = False

    def _cmd_is_active(self, msg: Twist) -> bool:
        return abs(msg.linear.x) > self.release_cmd_threshold or abs(msg.angular.z) > self.release_cmd_threshold

    def _recent_active_cmd(self) -> bool:
        if self.last_cmd_time is None:
            return False
        elapsed = (self.get_clock().now() - self.last_cmd_time).nanoseconds * 1e-9
        if elapsed > self.cmd_timeout:
            return False
        return self._cmd_is_active(self.last_cmd)

    def _on_cmd_msg(self, msg: Twist) -> None:
        now_t = self.get_clock().now()
        if self.same_topic_mode and self.last_guard_pub_time is not None:
            age = (now_t - self.last_guard_pub_time).nanoseconds * 1e-9
            if age <= self.same_topic_echo_window:
                if (
                    abs(msg.linear.x - self.last_guard_linear) <= self.same_topic_match_eps
                    and abs(msg.angular.z - self.last_guard_angular) <= self.same_topic_match_eps
                ):
                    # Ignore self-echo command to avoid treating guard output as an active user command.
                    return
        self.last_cmd = msg
        self.last_cmd_time = now_t
        if self._cmd_is_active(msg):
            self.idle_anchor_valid = False

    def _on_odom(self, msg: Odometry) -> None:
        self.vx = float(msg.twist.twist.linear.x)
        self.wz = float(msg.twist.twist.angular.z)
        self.px = float(msg.pose.pose.position.x)
        self.py = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)
        self.have_pose = True
        self.have_odom = True
        self.last_odom_time = self.get_clock().now()

    def _odom_is_fresh(self) -> bool:
        if not self.have_odom or self.last_odom_time is None:
            return False
        age = (self.get_clock().now() - self.last_odom_time).nanoseconds * 1e-9
        return age <= self.odom_timeout

    def _idle_guard_cmd(self) -> Twist:
        msg = Twist()
        if not self._odom_is_fresh():
            self.idle_anchor_valid = False
            return msg

        if not self.idle_anchor_valid:
            self._set_idle_anchor()

        linear_cmd = 0.0

        if abs(self.vx) > self.idle_brake_threshold:
            linear_cmd += -self.idle_linear_kp * self.vx

        if self.idle_anchor_valid and self.have_pose:
            dx = self.px - self.idle_anchor_x
            dy = self.py - self.idle_anchor_y
            forward_err = math.cos(self.idle_anchor_yaw) * dx + math.sin(self.idle_anchor_yaw) * dy
            if abs(forward_err) > self.idle_position_deadband:
                linear_cmd += -self.idle_position_kp * forward_err

        if abs(linear_cmd) > 0.0:
            if 0.0 < self.idle_min_linear_cmd and abs(linear_cmd) < self.idle_min_linear_cmd:
                linear_cmd = self.idle_min_linear_cmd if linear_cmd > 0.0 else -self.idle_min_linear_cmd
            msg.linear.x = self._clamp(
                linear_cmd,
                -self.idle_max_linear_cmd,
                self.idle_max_linear_cmd,
            )

        if abs(self.wz) > 0.01:
            msg.angular.z = self._clamp(
                -self.idle_angular_kp * self.wz,
                -self.idle_max_angular_cmd,
                self.idle_max_angular_cmd,
            )

        return msg

    def _active_cmd(self) -> Twist:
        out = Twist()
        target_linear = float(self.last_cmd.linear.x)
        out.linear.x = target_linear
        out.angular.z = float(self.last_cmd.angular.z)

        if self._odom_is_fresh():
            # Active tracking trims command to compensate model-induced speed bias.
            out.linear.x = self._clamp(
                target_linear + self.active_linear_kp * (target_linear - self.vx),
                -self.active_max_linear_cmd,
                self.active_max_linear_cmd,
            )
            if target_linear > self.release_cmd_threshold:
                out.linear.x = max(0.0, out.linear.x)
            elif target_linear < -self.release_cmd_threshold:
                out.linear.x = min(0.0, out.linear.x)

        if abs(target_linear) > self.release_cmd_threshold and abs(target_linear) < self.min_linear_cmd:
            out.linear.x = self.min_linear_cmd if target_linear > 0.0 else -self.min_linear_cmd

        if (
            self._odom_is_fresh()
            and abs(out.angular.z) <= self.straight_angular_deadband
            and abs(out.linear.x) > self.release_cmd_threshold
        ):
            correction = self._clamp(
                -self.yaw_damping_gain * self.wz,
                -self.yaw_correction_max,
                self.yaw_correction_max,
            )
            out.angular.z += correction

        return out

    def _on_timer(self) -> None:
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        if elapsed < self.hold_seconds:
            self._publish_cmd(0.0, 0.0)
            return

        if self.world_paused_by_me:
            self._resume_world_if_paused("Gazebo world resumed")

        if self._recent_active_cmd():
            # In same-topic mode, let external command publisher drive directly.
            # Do not republish active command, otherwise callback sees our own
            # messages and may keep stale commands alive.
            if self.same_topic_mode:
                return
            cmd = self._active_cmd()
        else:
            cmd = self._idle_guard_cmd()
        self._publish_cmd(cmd.linear.x, cmd.angular.z)

    def release_pause_if_needed(self) -> None:
        if self.world_paused_by_me:
            self._resume_world_if_paused("Gazebo world resumed on shutdown")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Startup hold + runtime drift guard for Gazebo vehicle")
    parser.add_argument("--cmd-topic", default="/car_0/teleop_cmd_vel", help="Input command topic")
    parser.add_argument("--drive-topic", default="/car_0/cmd_vel", help="Output drive topic")
    parser.add_argument("--drive-topic-reverse", default="", help="Optional reverse drive topic")
    parser.add_argument("--odom-topic", default="/car_0/wheel_odom", help="Odometry topic")
    parser.add_argument("--rate", type=float, default=20.0, help="Publish rate in Hz")
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.0,
        help="Initial duration to hold zero output before switching to runtime guard",
    )
    parser.add_argument("--kp", type=float, default=0.0, help="Idle linear damping gain")
    parser.add_argument("--max-cmd", type=float, default=0.0, help="Idle max linear correction magnitude")
    parser.add_argument("--speed-deadband", type=float, default=0.01, help="Ignore tiny odom speed (m/s)")
    parser.add_argument(
        "--idle-brake-threshold",
        type=float,
        default=0.01,
        help="Apply idle linear braking only when |vx| exceeds this threshold (m/s)",
    )
    parser.add_argument(
        "--idle-min-linear-cmd",
        type=float,
        default=0.0,
        help="Minimum |linear.x| braking command when idle brake is active",
    )
    parser.add_argument(
        "--release-cmd-threshold",
        type=float,
        default=0.02,
        help="Treat command as active when |linear.x| or |angular.z| exceeds this value",
    )
    parser.add_argument("--cmd-timeout", type=float, default=0.15, help="Active command timeout (s)")
    parser.add_argument("--odom-timeout", type=float, default=0.35, help="Treat odometry as stale after this timeout (s)")
    parser.add_argument("--min-linear-cmd", type=float, default=0.0, help="Optional minimum |linear.x| when command is active")
    parser.add_argument("--active-linear-kp", type=float, default=0.0, help="Active linear speed tracking gain")
    parser.add_argument("--active-max-linear-cmd", type=float, default=4.0, help="Clamp for active linear command output")
    parser.add_argument("--idle-angular-kp", type=float, default=0.0, help="Idle angular damping gain")
    parser.add_argument("--idle-max-angular-cmd", type=float, default=0.0, help="Idle max angular correction magnitude")
    parser.add_argument("--straight-angular-deadband", type=float, default=0.02, help="Treat small commanded angular.z as straight-line intent")
    parser.add_argument("--yaw-damping-gain", type=float, default=0.0, help="Yaw-rate damping gain during straight driving")
    parser.add_argument("--yaw-correction-max", type=float, default=0.0, help="Clamp for yaw correction during straight driving")
    parser.add_argument("--forward-cmd-scale-fwd", type=float, default=1.0, help="Forward command scale on forward drive topic")
    parser.add_argument("--forward-cmd-scale-rev", type=float, default=1.0, help="Forward command scale on reverse drive topic")
    parser.add_argument("--reverse-cmd-scale-fwd", type=float, default=0.0, help="Reverse command scale on forward drive topic")
    parser.add_argument("--reverse-cmd-scale-rev", type=float, default=2.5, help="Reverse command scale on reverse drive topic")
    parser.add_argument("--cmd-filter-alpha", type=float, default=0.2, help="Low-pass filter alpha for drive commands (0 disables, 1 no smoothing)")
    parser.add_argument(
        "--pause-world-during-hold",
        type=int,
        default=0,
        choices=[0, 1],
        help="Pause Gazebo physics during hold (1=true, 0=false)",
    )
    parser.add_argument("--publish-max-linear-cmd", type=float, default=2.5, help="Clamp for published drive linear.x")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rclpy.init()

    node = StartupSpeedHold(
        cmd_topic=args.cmd_topic,
        drive_topic=args.drive_topic,
        drive_topic_reverse=args.drive_topic_reverse,
        odom_topic=args.odom_topic,
        rate_hz=args.rate,
        hold_seconds=args.hold_seconds,
        kp=args.kp,
        max_cmd=args.max_cmd,
        speed_deadband=args.speed_deadband,
        idle_brake_threshold=args.idle_brake_threshold,
        idle_min_linear_cmd=args.idle_min_linear_cmd,
        release_cmd_threshold=args.release_cmd_threshold,
        cmd_timeout=args.cmd_timeout,
        odom_timeout=args.odom_timeout,
        min_linear_cmd=args.min_linear_cmd,
        active_linear_kp=args.active_linear_kp,
        active_max_linear_cmd=args.active_max_linear_cmd,
        idle_angular_kp=args.idle_angular_kp,
        idle_max_angular_cmd=args.idle_max_angular_cmd,
        straight_angular_deadband=args.straight_angular_deadband,
        yaw_damping_gain=args.yaw_damping_gain,
        yaw_correction_max=args.yaw_correction_max,
        forward_cmd_scale_fwd=args.forward_cmd_scale_fwd,
        forward_cmd_scale_rev=args.forward_cmd_scale_rev,
        reverse_cmd_scale_fwd=args.reverse_cmd_scale_fwd,
        reverse_cmd_scale_rev=args.reverse_cmd_scale_rev,
        cmd_filter_alpha=args.cmd_filter_alpha,
        pause_world_during_hold=bool(args.pause_world_during_hold),
        publish_max_linear_cmd=args.publish_max_linear_cmd,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.release_pause_if_needed()
        if rclpy.ok():
            rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
