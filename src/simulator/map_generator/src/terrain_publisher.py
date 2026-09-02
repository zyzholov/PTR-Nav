#!/usr/bin/env python3
"""
ROS2 Terrain Cloud Publisher - Pure Python Implementation
Loads Rocky Desert Mesh PCD and publishes as PointCloud2
No compiled C++ dependencies to avoid Boost library conflicts
"""

import sys
import struct
import time

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header
except ImportError as e:
    print(f"Error importing ROS2: {e}")
    print("Falling back to dummy publisher...")
    sys.exit(1)

import numpy as np
import os

try:
    from ament_index_python.packages import get_package_share_directory
except Exception:
    get_package_share_directory = None


class TerrainPublisherPython(Node):
    def __init__(self):
        super().__init__('terrain_publisher_python')
        
        # Parameters
        self.declare_parameter('pcd_file', 'RockyDesertMesh.pcd')
        self.declare_parameter('pub_rate', 10.0)
        self.declare_parameter('frame_id', 'world')
        
        pcd_file = self.get_parameter('pcd_file').value
        pub_rate = self.get_parameter('pub_rate').value
        self.frame_id = self.get_parameter('frame_id').value
        
        # Load terrain
        self.points = self.load_pcd(pcd_file)
        
        if self.points is not None:
            self.get_logger().info(
                f'Loaded terrain: {len(self.points)} points from {pcd_file}')
        else:
            self.get_logger().warn('Failed to load PCD, using synthetic data')
            self.points = self.generate_synthetic()
        
        # Publisher
        self.pub = self.create_publisher(PointCloud2, 'terrain_cloud', 10)
        
        # Timer
        self.timer = self.create_timer(1.0/pub_rate, self.publish_cloud)
        self.get_logger().info('Terrain Publisher Ready')
    
    def load_pcd(self, filename):
        """Load binary PCD file"""
        paths = [
            filename,
            f'./install/share/fake_ugv/maps/{filename}'
        ]

        if get_package_share_directory is not None:
            try:
                share_dir = get_package_share_directory('fake_ugv')
                paths.append(os.path.join(share_dir, 'maps', filename))
            except Exception:
                pass
        
        for path in paths:
            if os.path.exists(path):
                try:
                    return self._parse_pcd(path)
                except Exception as e:
                    self.get_logger().warn(f'Error loading {path}: {e}')
        
        return None
    
    def _parse_pcd(self, filepath):
        """Parse binary PCD format"""
        with open(filepath, 'rb') as f:
            # Read header
            num_points = 0
            offset = 0
            
            while True:
                line = f.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    break
                
                if line.startswith('POINTS'):
                    num_points = int(line.split()[-1])
                elif line.startswith('DATA'):
                    if 'binary' not in line:
                        raise ValueError('Only binary format supported')
                    break
            
            # Read binary data
            points = []
            for _ in range(num_points):
                data = f.read(12)
                if len(data) < 12:
                    break
                x, y, z = struct.unpack('<fff', data)
                points.append([x, y, z])
            
            return np.array(points, dtype=np.float32) if points else None
    
    def generate_synthetic(self):
        """Generate test terrain"""
        x = np.linspace(-25, 25, 100)
        y = np.linspace(-25, 25, 100)
        xx, yy = np.meshgrid(x, y)
        zz = 0.5 * np.sin(xx * 0.2) * np.cos(yy * 0.2) + 0.2
        return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)
    
    def publish_cloud(self):
        """Publish PointCloud2"""
        if self.points is None or len(self.points) == 0:
            return
        
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height = 1
        msg.width = len(self.points)
        msg.is_bigendian = False
        msg.is_dense = True
        
        # Fields: x, y, z as floats
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        
        # Convert points to bytes
        msg.data = b''.join(struct.pack('<fff', p[0], p[1], p[2]) for p in self.points)
        
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TerrainPublisherPython()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
