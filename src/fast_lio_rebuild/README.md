# FAST-LIO Rebuild Compare

This package is for replaying a LiDAR/IMU bag through FAST-LIO again while a
previous odometry bag is replayed at the same time.

Default topics:

- Raw bag input: `/livox/lidar`, `/imu/data_raw`
- Rebuilt odometry: `/fastlio_rebuild/Odometry`
- Rebuilt path: `/fastlio_rebuild/path`
- Rebuilt registered cloud: `/fastlio_rebuild/cloud_registered`
- Reference odometry from old bag: `/fastlio_reference/Odometry`
- Sliding guide path: `/fastlio_rebuild/reference_window_path`
- RViz styled guide marker: `/fastlio_rebuild/reference_window_marker`

The reference odometry bag should be started a little before the raw bag, or
`raw_bag_delay` should be slightly larger than `reference_bag_delay`. The
default launch delay starts the reference bag after 0.01 s and the raw bag after
0.60 s, while the guide node uses a 0.45 s reference lead time.
