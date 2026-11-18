#!/usr/bin/env python3
"""Qualification-only risk-message dropout. Never pause a CUDA process."""
import rclpy
from rclpy.node import Node
from se2_grid_msgs.msg import SE2Grid
from std_srvs.srv import SetBool


class MapFaultRelay(Node):
    def __init__(self):
        super().__init__("qualification_map_fault_relay")
        self.paused=False
        self.pub=self.create_publisher(SE2Grid,"/qualification/fused_map",1)
        self.create_subscription(SE2Grid,"/car_0_fsgp_bgk/fused_map",self.forward,1)
        self.create_service(SetBool,"/qualification/pause_map_relay",self.pause)

    def forward(self,msg):
        if not self.paused:self.pub.publish(msg)

    def pause(self,request,response):
        self.paused=request.data;response.success=True
        response.message="dropping maps" if self.paused else "forwarding fresh maps, no replay"
        return response


def main():
    rclpy.init();node=MapFaultRelay()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=="__main__":main()
