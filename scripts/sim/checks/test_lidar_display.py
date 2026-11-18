from pathlib import Path
import os
import signal
import subprocess
import sys
import time

import numpy as np
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32,read_points
from std_msgs.msg import Header


def test_actual_display_binary_preserves_all_xyz_and_stamps_and_changes_only_colors():
    # Run the DDS lifecycle in a fresh process, as in the real launch. Other
    # unit tests repeatedly initialize/shut down ROS while holding mock nodes.
    process=subprocess.Popen([sys.executable,str(Path(__file__)),'--isolated'],
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
    try:
        output,_=process.communicate(timeout=25)
        assert process.returncode==0,output
    finally:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGINT)
            try:process.communicate(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.communicate()


def qualify_display_binary():
    root=Path(__file__).resolve().parents[3]
    binary=root/'install/se2_grid_converter/lib/se2_grid_converter/lidar_display_node'
    assert binary.is_file(), 'build the actual presentation binary before testing'
    prefix=f'/test_lidar_display_{os.getpid()}'
    topics={name:prefix+'/'+name for name in ('raw_map','raw_live','odom','map','live')}
    remaps=[]
    for source,name in [('/Laser_map','raw_map'),('/cloud_registered_visual','raw_live'),
                        ('/Odometry','odom'),('/terrain_display/lidar_map','map'),
                        ('/terrain_display/lidar_returns','live')]:
        remaps.extend(['-r',source+':='+topics[name]])
    process=subprocess.Popen([str(binary),'--ros-args',*remaps],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    context=Context();rclpy.init(args=[],context=context)
    node=rclpy.create_node('lidar_presentation_qualification',context=context)
    executor=SingleThreadedExecutor(context=context);executor.add_node(node)
    sources={name:node.create_publisher(PointCloud2,topic,1) for name,topic in
             [('map',topics['raw_map']),('live',topics['raw_live'])]}
    odometry=node.create_publisher(Odometry,topics['odom'],1)
    received={};live_events=[]
    def record(name,msg):
        received[name]=msg
        if name=='live':live_events.append(((msg.header.stamp.sec,msg.header.stamp.nanosec),msg.width*msg.height))
    subscriptions=[node.create_subscription(PointCloud2,topic,lambda msg,n=name:record(n,msg),1)
        for name,topic in [('map',topics['map']),('live',topics['live'])]]
    xyz=np.array([[1,0,0],[8,0,1],[25,0,2],[2,3,-.2],[np.nan,1,2]],dtype=np.float32)
    expected={}
    try:
        deadline=time.monotonic()+8
        while time.monotonic()<deadline and len(received)<2:
            stamp=node.get_clock().now().to_msg()
            odom=Odometry();odom.header=Header(frame_id='map',stamp=stamp)
            odom.pose.pose.orientation.w=1.;odometry.publish(odom)
            for name,pub in sources.items():
                cloud=create_cloud_xyz32(Header(frame_id='map',stamp=stamp),xyz)
                expected[(stamp.sec,stamp.nanosec)]=xyz.copy();pub.publish(cloud)
            until=time.monotonic()+.05
            while time.monotonic()<until:executor.spin_once(timeout_sec=.01)
            assert process.poll() is None
        assert set(received)=={'map','live'}
        for msg in received.values():
            key=(msg.header.stamp.sec,msg.header.stamp.nanosec)
            assert key in expected and msg.header.frame_id=='map'
            rows=read_points(msg,field_names=('x','y','z','rgb'))
            actual=np.column_stack([rows[n] for n in ('x','y','z')])
            np.testing.assert_array_equal(actual,expected[key])
            assert len(np.unique(rows['rgb'].view(np.uint32)))>=3
        # Fresh data in a wrong frame cannot replace a valid display generation.
        old=received['live'].header.stamp
        bad=create_cloud_xyz32(Header(frame_id='wrong',stamp=node.get_clock().now().to_msg()),xyz)
        sources['live'].publish(bad)
        until=time.monotonic()+.2
        while time.monotonic()<until:executor.spin_once(timeout_sec=.01)
        assert received['live'].header.stamp==old
        # The renderer holds one scan between arrivals (decay=0), then the
        # publisher clears it once on actual staleness. No restamped replay.
        until=time.monotonic()+.8
        while time.monotonic()<until:executor.spin_once(timeout_sec=.01)
        assert received['live'].width*received['live'].height==0
        assert received['live'].header.stamp==old
        assert sum(count==0 for key,count in live_events)==1
        stamp=node.get_clock().now().to_msg()
        pose=Odometry();pose.header=Header(frame_id='map',stamp=stamp);pose.pose.pose.orientation.w=1.
        odometry.publish(pose)
        fresh=create_cloud_xyz32(Header(frame_id='map',stamp=stamp),xyz);sources['live'].publish(fresh)
        until=time.monotonic()+.2
        while time.monotonic()<until and received['live'].header.stamp!=stamp:executor.spin_once(timeout_sec=.01)
        assert received['live'].header.stamp==stamp and received['live'].width*received['live'].height==len(xyz)
        from rclpy.qos import ReliabilityPolicy
        endpoint=next(info for info in node.get_subscriptions_info_by_topic(topics['raw_live']) if info.node_name=='lidar_display')
        assert endpoint.qos_profile.reliability==ReliabilityPolicy.RELIABLE
        for subscription in subscriptions:node.destroy_subscription(subscription)
        deadline=time.monotonic()+3
        while time.monotonic()<deadline and sources['live'].get_subscription_count():
            executor.spin_once(timeout_sec=.05)
        assert sources['live'].get_subscription_count()==0, (
            'dense scan demand must disappear when displays close',
            [info.node_name for info in node.get_subscriptions_info_by_topic(topics['raw_live'])],
            [info.node_name for info in node.get_subscriptions_info_by_topic(topics['live'])])
    finally:
        del subscriptions
        executor.remove_node(node);executor.shutdown()
        node.destroy_node();rclpy.shutdown(context=context)
        process.send_signal(signal.SIGINT)
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=='__main__':
    assert sys.argv[1:]==['--isolated']
    qualify_display_binary()
