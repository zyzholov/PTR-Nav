#!/usr/bin/env python3
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterUninitializedException
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosidl_runtime_py.utilities import get_message
import tf2_ros


@dataclass
class TopicState:
    topic_type: Optional[str] = None
    count: int = 0
    last_count: int = 0
    last_seen: float = 0.0
    subscription: object = None


class Livo2ChainCheck(Node):
    def __init__(self):
        super().__init__("livo2_chain_check")
        self.declare_parameter("required_topics", [
            "/livox/lidar",
            "/livox/imu",
            "/camera/camera/color/image_raw",
            "/Odometry",
            "/cloud_registered",
        ])
        self.declare_parameter("optional_topics", [
            "/path",
            "/rgb_img",
            "/camera/camera/depth/image_rect_raw",
            "/car_0_terrain_analyzer/fused_map",
            "/car_0_terrain_analyzer/sdf_map",
        ])
        self.declare_parameter("tf_checks", [
            "map,camera_init",
            "map,aft_mapped",
        ])
        self.declare_parameter("report_period_sec", 2.0)
        self.declare_parameter("warn_after_sec", 8.0)

        self.required_topics = self._string_list_parameter("required_topics")
        self.optional_topics = self._string_list_parameter("optional_topics")
        self.tf_checks = self._string_list_parameter("tf_checks")
        self.report_period_sec = float(self.get_parameter("report_period_sec").value)
        self.warn_after_sec = float(self.get_parameter("warn_after_sec").value)
        self.start_time = time.monotonic()
        self.states: Dict[str, TopicState] = {
            topic: TopicState() for topic in self.required_topics + self.optional_topics
        }
        self.qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_timer(0.5, self._discover_topics)
        self.create_timer(self.report_period_sec, self._report)

    def _string_list_parameter(self, name: str) -> List[str]:
        try:
            value = self.get_parameter(name).value
        except ParameterUninitializedException:
            return []
        if value is None:
            return []
        return list(value)

    def _discover_topics(self):
        topic_types = {}
        for topic, types in self.get_topic_names_and_types():
            if types:
                topic_types[topic] = types[0]

        for topic, state in self.states.items():
            if state.subscription is not None:
                continue
            topic_type = topic_types.get(topic)
            if not topic_type:
                continue
            try:
                msg_type = get_message(topic_type)
            except (AttributeError, ModuleNotFoundError, ValueError) as exc:
                self.get_logger().warn(f"{topic}: cannot import {topic_type}: {exc}")
                continue
            state.topic_type = topic_type
            state.subscription = self.create_subscription(
                msg_type,
                topic,
                lambda _msg, t=topic: self._mark_seen(t),
                self.qos,
            )
            self.get_logger().info(f"watching {topic} [{topic_type}]")

    def _mark_seen(self, topic: str):
        state = self.states[topic]
        state.count += 1
        state.last_seen = time.monotonic()

    def _topic_status(self, topic: str) -> str:
        state = self.states[topic]
        if state.subscription is None:
            return "missing"
        hz = (state.count - state.last_count) / max(self.report_period_sec, 1e-6)
        age = time.monotonic() - state.last_seen if state.last_seen else float("inf")
        state.last_count = state.count
        if state.count == 0:
            return "waiting"
        if age > self.warn_after_sec:
            return f"stale age={age:.1f}s"
        return f"ok {hz:.1f}Hz"

    def _tf_status(self, spec: str) -> str:
        parts = [p.strip() for p in spec.split(",")]
        if len(parts) != 2:
            return "bad spec"
        target, source = parts
        try:
            self.tf_buffer.lookup_transform(target, source, rclpy.time.Time())
            return "ok"
        except Exception as exc:
            return f"missing ({exc.__class__.__name__})"

    def _report(self):
        elapsed = time.monotonic() - self.start_time
        required = [f"{t}={self._topic_status(t)}" for t in self.required_topics]
        optional = [f"{t}={self._topic_status(t)}" for t in self.optional_topics]
        tf_status = [f"{s}={self._tf_status(s)}" for s in self.tf_checks]

        required_ok = all(self.states[t].count > 0 for t in self.required_topics)
        if required_ok:
            self.get_logger().info("required chain OK | " + " | ".join(required))
        elif elapsed > self.warn_after_sec:
            self.get_logger().warn("required chain incomplete | " + " | ".join(required))
        else:
            self.get_logger().info("waiting for chain | " + " | ".join(required))

        self.get_logger().info("optional | " + " | ".join(optional))
        self.get_logger().info("tf | " + " | ".join(tf_status))


def main():
    rclpy.init()
    node = None
    try:
        node = Livo2ChainCheck()
        rclpy.spin(node)
    except (KeyboardInterrupt, RuntimeError):
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
