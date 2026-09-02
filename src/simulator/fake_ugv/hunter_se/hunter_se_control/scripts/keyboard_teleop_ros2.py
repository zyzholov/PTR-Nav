#!/usr/bin/python3

import argparse
import select
import sys
import termios
import tty

import rclpy
from rclpy.utilities import remove_ros_args
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


HELP = """
Hunter SE keyboard teleop

Controls:
  w / s : increase / decrease target linear speed
  a / d : increase / decrease latched angular.z steering command
  space : full stop
  x     : zero angular.z steering command
  q / e : decrease / increase max linear speed
  z / c : decrease / increase max angular.z steering command
  h     : print help
  Ctrl-C: quit
"""


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def step_with_zero_snap(current: float, delta: float) -> float:
    updated = current + delta
    if current != 0.0 and current * updated < 0.0:
        return 0.0
    return updated


class HunterKeyboard(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("hunter_se_keyboard_teleop")
        self.publisher = self.create_publisher(Twist, args.topic, 10)
        self.odom_speed = 0.0
        self.target_linear = 0.0
        self.target_yaw_rate = 0.0
        self.max_linear = args.max_linear
        self.max_yaw_rate = args.max_yaw_rate
        self.step_linear = args.step_linear
        self.step_yaw_rate = args.step_yaw_rate
        self.center_yaw_on_speed_step = args.center_yaw_on_speed_step
        self.hold_to_run = args.hold_to_run
        self.hold_timeout = max(0.05, args.hold_timeout)
        self.yaw_hold_timeout = max(0.0, args.yaw_hold_timeout)
        self.speed_kp = max(0.0, args.speed_kp)
        self.publish_rate = max(5.0, args.rate)
        self.command_topic = args.topic
        self._last_multi_pub_warn_ns = 0
        self.last_yaw_input_time = self.get_clock().now()
        self.last_motion_input_time = self.get_clock().now()
        self.create_subscription(Odometry, args.odom_topic, self._odom_cb, 10)
        self.create_timer(1.0 / self.publish_rate, self._publish)
        self.create_timer(2.0, self._warn_if_other_publishers)
        self.get_logger().info(f"Publishing Hunter SE Twist commands to {args.topic}")

    def _odom_cb(self, msg: Odometry) -> None:
        self.odom_speed = float(msg.twist.twist.linear.x)

    def handle_key(self, key: str) -> None:
        if key == "w":
            self.target_linear = self.max_linear if self.hold_to_run else step_with_zero_snap(self.target_linear, self.step_linear)
            if self.center_yaw_on_speed_step and not self.hold_to_run:
                self.target_yaw_rate = 0.0
            self.last_motion_input_time = self.get_clock().now()
        elif key == "s":
            self.target_linear = -self.max_linear if self.hold_to_run else step_with_zero_snap(self.target_linear, -self.step_linear)
            if self.center_yaw_on_speed_step and not self.hold_to_run:
                self.target_yaw_rate = 0.0
            self.last_motion_input_time = self.get_clock().now()
        elif key == "a":
            self.target_yaw_rate = self.max_yaw_rate if self.hold_to_run else step_with_zero_snap(self.target_yaw_rate, self.step_yaw_rate)
            self.last_yaw_input_time = self.get_clock().now()
            self.last_motion_input_time = self.get_clock().now()
        elif key == "d":
            self.target_yaw_rate = -self.max_yaw_rate if self.hold_to_run else step_with_zero_snap(self.target_yaw_rate, -self.step_yaw_rate)
            self.last_yaw_input_time = self.get_clock().now()
            self.last_motion_input_time = self.get_clock().now()
        elif key == " ":
            self.target_linear = 0.0
            self.target_yaw_rate = 0.0
        elif key == "x":
            self.target_yaw_rate = 0.0
        elif key == "q":
            self.max_linear = max(0.2, self.max_linear - 0.1)
        elif key == "e":
            self.max_linear += 0.1
        elif key == "z":
            self.max_yaw_rate = max(0.1, self.max_yaw_rate - 0.05)
        elif key == "c":
            self.max_yaw_rate += 0.05
        elif key == "h":
            print(HELP)
            return
        else:
            return

        self.target_linear = clamp(self.target_linear, -self.max_linear, self.max_linear)
        self.target_yaw_rate = clamp(self.target_yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)
        self._publish()
        self._print_status()

    def _publish(self) -> None:
        msg = Twist()
        if self.hold_to_run:
            motion_idle_sec = (self.get_clock().now() - self.last_motion_input_time).nanoseconds * 1e-9
            if motion_idle_sec > self.hold_timeout:
                self.target_linear = 0.0
                self.target_yaw_rate = 0.0
        if self.yaw_hold_timeout > 0.0 and abs(self.target_yaw_rate) > 1e-6:
            yaw_idle_sec = (self.get_clock().now() - self.last_yaw_input_time).nanoseconds * 1e-9
            if yaw_idle_sec > self.yaw_hold_timeout:
                self.target_yaw_rate = 0.0
        speed_error = self.target_linear - self.odom_speed
        msg.linear.x = float(clamp(self.target_linear + self.speed_kp * speed_error, -self.max_linear, self.max_linear))
        msg.angular.z = float(self.target_yaw_rate)
        self.publisher.publish(msg)

    def _warn_if_other_publishers(self) -> None:
        try:
            infos = self.get_publishers_info_by_topic(self.command_topic)
        except Exception:
            return

        if len(infos) <= 1:
            return

        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_multi_pub_warn_ns < int(2e9):
            return

        self._last_multi_pub_warn_ns = now_ns
        self.get_logger().warn(
            f"Detected {len(infos)} publishers on {self.command_topic}. "
            "Unexpected angular motion while using keyboard teleop is usually caused by another controller still publishing cmd_vel."
        )

    def _print_status(self) -> None:
        print(
            f"target_linear={self.target_linear:+.2f} m/s "
            f"measured_linear={self.odom_speed:+.2f} m/s "
            f"angular_z_steer_cmd={self.target_yaw_rate:+.2f} "
            f"(max_linear={self.max_linear:.2f}, max_angular_z={self.max_yaw_rate:.2f})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hunter SE ROS 2 keyboard teleop")
    parser.add_argument("--topic", default="/hunter_se/cmd_vel", help="Twist command topic")
    parser.add_argument("--odom-topic", default="/hunter_se/wheel_odom", help="Odometry topic")
    parser.add_argument("--rate", type=float, default=50.0, help="Publish rate in Hz")
    parser.add_argument("--max-linear", type=float, default=1.5, help="Maximum target linear speed")
    parser.add_argument("--max-yaw-rate", "--max-steer", dest="max_yaw_rate", type=float, default=1.20,
                        help="Maximum |angular.z| command; in Isaac/Gazebo Hunter this is the steering angle")
    parser.add_argument("--step-linear", type=float, default=0.10, help="Linear speed increment")
    parser.add_argument("--step-yaw-rate", "--step-steer", dest="step_yaw_rate", type=float, default=0.08,
                        help="Increment applied to Twist.angular.z")
    parser.add_argument("--yaw-hold-timeout", type=float, default=0.0,
                        help="Auto-center angular.z after no angular key input for this duration; 0 keeps latched command")
    parser.add_argument("--hold-to-run", action="store_true",
                        help="Treat movement keys as hold commands and publish zero after --hold-timeout with no key input")
    parser.add_argument("--hold-timeout", type=float, default=0.25,
                        help="Seconds without motion key input before hold-to-run publishes a full stop")
    parser.add_argument("--center-steer-on-speed-step", "--center-yaw-on-speed-step",
                        action="store_true", dest="center_yaw_on_speed_step", default=False,
                        help="Reset angular.z to zero when changing speed")
    parser.add_argument("--no-center-steer-on-speed-step", "--no-center-yaw-on-speed-step",
                        action="store_false", dest="center_yaw_on_speed_step",
                        help="Keep the previous angular.z value when changing speed")
    parser.add_argument("--speed-kp", type=float, default=0.0, help="Simple odom feedback gain")
    cli_args = remove_ros_args(args=sys.argv)[1:]
    return parser.parse_args(cli_args)


def read_key(timeout_s: float) -> str:
    ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not ready:
        return ""
    char = sys.stdin.read(1)
    if char != "\x1b":
        return char
    sequence = sys.stdin.read(2)
    if sequence == "[A":
        return "w"
    if sequence == "[B":
        return "s"
    if sequence == "[C":
        return "d"
    if sequence == "[D":
        return "a"
    return ""


def main() -> int:
    args = parse_args()
    print(HELP)
    rclpy.init()
    node = HunterKeyboard(args)

    settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            key = read_key(0.05)
            if key == "\x03":
                break
            if key:
                node.handle_key(key)
    except KeyboardInterrupt:
        pass
    finally:
        stop_msg = Twist()
        if rclpy.ok():
            try:
                node.publisher.publish(stop_msg)
            except Exception:
                pass
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
