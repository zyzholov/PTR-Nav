#!/usr/bin/env python3
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.serialization import deserialize_message

import rosbag2_py


@dataclass
class BagOdom:
    record_time_s: float
    msg: Odometry


class OdomBagRepublisher(Node):
    """Replay an odometry topic from a bag to a new topic without ros2 bag remap."""

    def __init__(self) -> None:
        super().__init__("odom_bag_republisher")
        self.declare_parameter("bag_path", "/home/z/workspace_zyz/bags/rosbag2_2026_07_07-21_53_43")
        self.declare_parameter("input_topic", "/Odometry")
        self.declare_parameter("output_topic", "/fastlio_reference/Odometry")
        self.declare_parameter("play_rate", 1.0)
        self.declare_parameter("start_delay", 0.01)
        self.declare_parameter("start_offset", 0.0)
        self.declare_parameter("loop", False)
        self.declare_parameter("publish_period", 0.002)

        self.bag_path = str(self.get_parameter("bag_path").value)
        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.play_rate = max(1e-6, float(self.get_parameter("play_rate").value))
        self.start_delay = max(0.0, float(self.get_parameter("start_delay").value))
        self.start_offset = max(0.0, float(self.get_parameter("start_offset").value))
        self.loop = bool(self.get_parameter("loop").value)
        publish_period = max(0.001, float(self.get_parameter("publish_period").value))

        self.messages = self._load_messages()
        self.publisher = self.create_publisher(Odometry, self.output_topic, 100)
        self.first_record_s = self.messages[0].record_time_s if self.messages else 0.0
        self.last_record_s = self.messages[-1].record_time_s if self.messages else 0.0
        self.index = 0
        self.total_published = 0
        self.last_status_wall_s = 0.0
        self.wall_start_s = time.monotonic() + self.start_delay
        self._apply_start_offset()
        self.timer = self.create_timer(publish_period, self._timer_callback)

        self.get_logger().info(
            "Odom bag republisher: %s %s -> %s, msgs=%d, delay=%.3fs, rate=%.2f, start_offset=%.2fs"
            % (
                self.bag_path,
                self.input_topic,
                self.output_topic,
                len(self.messages),
                self.start_delay,
                self.play_rate,
                self.start_offset,
            )
        )

    def _load_messages(self) -> List[BagOdom]:
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id="sqlite3")
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
        if self.input_topic not in topic_types:
            raise RuntimeError(
                "Input topic %s not found in bag %s. Available: %s"
                % (self.input_topic, self.bag_path, sorted(topic_types.keys()))
            )
        if topic_types[self.input_topic] != "nav_msgs/msg/Odometry":
            raise RuntimeError(
                "Input topic %s has type %s, expected nav_msgs/msg/Odometry"
                % (self.input_topic, topic_types[self.input_topic])
            )

        messages: List[BagOdom] = []
        while reader.has_next():
            topic, data, record_time_ns = reader.read_next()
            if topic != self.input_topic:
                continue
            msg = deserialize_message(data, Odometry)
            messages.append(BagOdom(record_time_s=float(record_time_ns) * 1e-9, msg=msg))
        if not messages:
            raise RuntimeError("No odometry messages read from %s:%s" % (self.bag_path, self.input_topic))
        return messages

    def _apply_start_offset(self) -> None:
        if self.start_offset <= 0.0:
            return
        target_record_s = self.first_record_s + self.start_offset
        while self.index < len(self.messages) and self.messages[self.index].record_time_s < target_record_s:
            self.index += 1
        self.wall_start_s = time.monotonic() + self.start_delay
        if self.index < len(self.messages):
            self.first_record_s = self.messages[self.index].record_time_s

    def _timer_callback(self) -> None:
        if not self.messages:
            return
        now_s = time.monotonic()
        if now_s < self.wall_start_s:
            return
        elapsed_bag_s = (now_s - self.wall_start_s) * self.play_rate
        target_record_s = self.first_record_s + elapsed_bag_s

        published = 0
        while self.index < len(self.messages) and self.messages[self.index].record_time_s <= target_record_s:
            self.publisher.publish(self.messages[self.index].msg)
            self.index += 1
            published += 1
            self.total_published += 1
            if published >= 20:
                break

        if self.total_published > 0 and now_s - self.last_status_wall_s > 2.0:
            self.last_status_wall_s = now_s
            self.get_logger().info(
                "Published reference odom: %d/%d to %s"
                % (self.total_published, len(self.messages), self.output_topic)
            )

        if self.index >= len(self.messages):
            if self.loop:
                self.index = 0
                self.first_record_s = self.messages[0].record_time_s
                self.wall_start_s = time.monotonic() + self.start_delay
            else:
                self.get_logger().info("Finished odometry bag replay.")
                self.timer.cancel()


def main(args: Optional[Iterable[str]] = None) -> None:
    rclpy.init(args=args)
    node = OdomBagRepublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
