#!/usr/bin/python3

import argparse
import select
import sys
import termios
import tty
from typing import Tuple

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


HELP = """
ROS2 keyboard teleop for Gazebo Ackermann (/car_0/cmd_vel)

Controls:
  w/s : linear speed + / -
  a/d : steer left / right (angular.z + / -)
  space: stop (linear and steering to zero)
  x    : center steering only (angular.z = 0)
  q/e  : decrease / increase max linear speed
  z/c  : decrease / increase max angular speed
  h    : print this help
  Ctrl-C : quit
"""


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class KeyboardTeleop(Node):
    def __init__(
        self,
        topic: str,
        odom_topic: str,
        publish_rate: float,
        max_linear: float,
        max_angular: float,
        step_linear: float,
        step_angular: float,
        idle_timeout: float,
        speed_kp: float,
        cmd_speed_scale: float,
    ) -> None:
        super().__init__("keyboard_ctrl")
        self.pub = self.create_publisher(Twist, topic, 10)
        self.timer = self.create_timer(1.0 / publish_rate, self._publish)

        self.max_linear = max_linear
        self.max_angular = max_angular
        self.step_linear = step_linear
        self.step_angular = step_angular
        self.idle_timeout = idle_timeout
        self.speed_kp = max(0.0, speed_kp)
        self.cmd_speed_scale = max(0.2, float(cmd_speed_scale))
        self.odom_speed_x = 0.0

        if self.speed_kp > 0.0:
            self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
            self.get_logger().info(
                f"Speed feedback enabled (kp={self.speed_kp:.2f}) from {odom_topic}"
            )

        self.target_linear = 0.0
        self.target_angular = 0.0
        self.last_input_time = self.get_clock().now()

        self.get_logger().info(f"Publishing Twist to {topic}")
        self._print_status()

    def _odom_cb(self, msg: Odometry) -> None:
        self.odom_speed_x = float(msg.twist.twist.linear.x)

    def _compute_linear_cmd(self) -> float:
        raw_cmd = self.target_linear
        if self.speed_kp <= 0.0:
            return clamp(raw_cmd * self.cmd_speed_scale, -self.max_linear * self.cmd_speed_scale, self.max_linear * self.cmd_speed_scale)

        speed_err = self.target_linear - self.odom_speed_x
        raw_cmd = clamp(
            self.target_linear + self.speed_kp * speed_err,
            -self.max_linear,
            self.max_linear,
        )
        return clamp(raw_cmd * self.cmd_speed_scale, -self.max_linear * self.cmd_speed_scale, self.max_linear * self.cmd_speed_scale)

    def handle_key(self, key: str) -> None:
        now = self.get_clock().now()
        changed = False

        if key == "w":
            self.target_linear += self.step_linear
            changed = True
        elif key == "s":
            self.target_linear -= self.step_linear
            changed = True
        elif key == "a":
            self.target_angular += self.step_angular
            changed = True
        elif key == "d":
            self.target_angular -= self.step_angular
            changed = True
        elif key == " ":
            self.target_linear = 0.0
            self.target_angular = 0.0
            changed = True
        elif key == "x":
            self.target_angular = 0.0
            changed = True
        elif key == "q":
            self.max_linear = max(0.1, self.max_linear - 0.1)
            changed = True
        elif key == "e":
            self.max_linear = self.max_linear + 0.1
            changed = True
        elif key == "z":
            self.max_angular = max(0.1, self.max_angular - 0.1)
            changed = True
        elif key == "c":
            self.max_angular = self.max_angular + 0.1
            changed = True
        elif key == "h":
            print(HELP)
            return
        else:
            return

        self.target_linear = clamp(self.target_linear, -self.max_linear, self.max_linear)
        self.target_angular = clamp(self.target_angular, -self.max_angular, self.max_angular)

        if changed:
            self.last_input_time = now
            self._print_status()

    def _publish(self) -> None:
        if self.idle_timeout > 0.0:
            dt = (self.get_clock().now() - self.last_input_time).nanoseconds * 1e-9
            if dt >= self.idle_timeout:
                self.target_linear = 0.0
                self.target_angular = 0.0

        msg = Twist()
        msg.linear.x = float(self._compute_linear_cmd())
        msg.angular.z = float(self.target_angular)
        self.pub.publish(msg)

    def _print_status(self) -> None:
        cmd_linear = self._compute_linear_cmd()
        print(
            f"target_linear={self.target_linear:+.2f} m/s  cmd_linear={cmd_linear:+.2f} m/s"
            f"  measured_linear={self.odom_speed_x:+.2f} m/s  angular={self.target_angular:+.2f} rad/s"
            f"  (max_linear={self.max_linear:.2f}, max_angular={self.max_angular:.2f}, speed_kp={self.speed_kp:.2f}, cmd_scale={self.cmd_speed_scale:.2f})"
        )


def _read_key(timeout_s: float) -> str:
    rlist, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not rlist:
        return ""

    ch = sys.stdin.read(1)
    if ch != "\x1b":
        return ch

    # Parse arrow keys: ESC [ A/B/C/D -> w/s/d/a mapping.
    seq = sys.stdin.read(2)
    if seq == "[A":
        return "w"
    if seq == "[B":
        return "s"
    if seq == "[C":
        return "d"
    if seq == "[D":
        return "a"
    return ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROS2 keyboard teleop for Gazebo vehicle")
    parser.add_argument("--topic", default="/car_0/cmd_vel", help="Twist topic")
    parser.add_argument("--odom-topic", default="/car_0/wheel_odom", help="Odometry topic")
    parser.add_argument("--rate", type=float, default=20.0, help="Publish rate in Hz")
    parser.add_argument("--max-linear", type=float, default=1.2, help="Max desired |linear.x| (m/s)")
    parser.add_argument("--max-angular", type=float, default=1.2, help="Max |angular.z| (rad/s)")
    parser.add_argument("--step-linear", type=float, default=0.20, help="Linear step per key")
    parser.add_argument("--step-angular", type=float, default=0.10, help="Angular step per key")
    parser.add_argument(
        "--speed-kp",
        type=float,
        default=0.8,
        help="Speed feedback gain using odom linear.x (0 disables)",
    )
    parser.add_argument(
        "--cmd-speed-scale",
        type=float,
        default=2.2,
        help="Scale factor from desired speed to published cmd_vel.linear.x",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=0.0,
        help="Auto-stop after no key input for this duration (s), 0 disables",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rclpy.init()

    node = KeyboardTeleop(
        topic=args.topic,
        odom_topic=args.odom_topic,
        publish_rate=args.rate,
        max_linear=args.max_linear,
        max_angular=args.max_angular,
        step_linear=args.step_linear,
        step_angular=args.step_angular,
        idle_timeout=args.idle_timeout,
        speed_kp=args.speed_kp,
        cmd_speed_scale=args.cmd_speed_scale,
    )

    print(HELP)

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            key = _read_key(0.05)
            if key == "\x03":
                break
            if key:
                node.handle_key(key)
            rclpy.spin_once(node, timeout_sec=0.0)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
