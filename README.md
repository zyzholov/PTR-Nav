# PTR-Nav

PTR-Nav 是一个面向复杂非结构化地形的 ROS 2 UGV 导航、建图、仿真与规划控制工程。当前仓库分支用于环境搭建和仿真链路前期整理，重点保留可复现的源码、配置、模型和脚本，不提交 `build/`、`install/`、`log/` 等本地编译/运行产物。

## 当前分支状态

本分支建议作为环境搭建前期分支使用，例如 `env-setup-prep`。主分支暂不切换到这套内容，等 Isaac Sim、FAST-LIO、Gazebo、规划控制链路稳定后再合并。

当前重点链路：

- ROS 2 Humble 工作空间，使用 `colcon` 构建。
- FAST-LIO / FAST-LIVO2 相关源码和 Hunter/MID360/D455 仿真配置。
- Gazebo Hunter SE + Livox MID360 仿真链路。
- Isaac Sim Hunter SE + MID360 RTX LiDAR + 外置 IMU + RealSense D455 仿真链路。
- 地形分析、SE2 栅格、Kino-A* / MPPI / MPC 等规划控制模块。

## 目录概览

- `src/FAST_LIO`: FAST-LIO ROS 2 版本及 Hunter/Gazebo/Isaac 配置。
- `src/FAST-LIVO2-ROS2`: FAST-LIVO2 ROS 2 版本及 D455 相关配置，当前保留作后续多传感器融合验证。
- `src/FSGP-BGK-ROS2`: FSGP-BGK 概率地形可通行性算法源码，当前作为待封装集成的第三方功能包。
- `src/isaac_sim_hunter_livo2`: Isaac Sim 场景脚本、传感器配置、Mars Gale 地形渲染与诊断脚本。
- `src/simulator`: Gazebo/Isaac 使用的 UGV、Livox、地图和仿真资源。
- `src/planner`: 地形导航、轨迹规划、控制和相关 launch 文件。
- `src/uneven_mapping`: 地形分析与 SE2 栅格工具。
- `docs`: 集成说明、部署说明和调试记录。

## 依赖准备

基础环境：

- Ubuntu 22.04
- ROS 2 Humble
- `colcon`
- Git LFS
- PCL / Eigen / OpenCV / Ceres 等 FAST-LIO/FAST-LIVO2 常用依赖
- Gazebo 11
- Isaac Sim 5.1，默认路径按本机脚本假设为 `/home/z/workspace_zyz/Isaac_Sim`

首次克隆后需要拉取 LFS 资产：

```bash
git lfs install
git lfs pull
```

ROS 2 依赖可按工程实际包继续补齐：

```bash
source /opt/ros/humble/setup.bash
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

## 构建

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

如果只调 FAST-LIO：

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select fast_lio --symlink-install
source install/setup.bash
```

## Gazebo FAST-LIO 链路

启动 Hunter + Gazebo 仿真：

```bash
./run_hunter_gazebo.sh
```

启动 FAST-LIO：

```bash
./run_hunter_fastlio.sh
```

`run_hunter_fastlio.sh` 会根据 `/livox/lidar` 的消息类型自动选择配置：

- `sensor_msgs/msg/PointCloud2`: `src/FAST_LIO/config/isaac_hunter_mid360_pointcloud.yaml`
- `livox_ros_driver2/msg/CustomMsg`: `src/FAST_LIO/config/gazebo_hunter_mid360.yaml`

## Isaac Sim FAST-LIO 链路

可视化仿真、传感器话题、键盘控制和 FAST-LIO 建图建议分终端启动。

Terminal 1，启动 Isaac Sim：

```bash
./run_hunter_isaac_sim.sh mapping:=true --scene-profile mars_gale --drive-mode cmd_vel --cmd-vel-angular-mode steer --mars-visual-detail full --mars-surface-skin full --sensor-visual-detail cad --viewer-camera close --obstacle-density sparse --camera-width 1280 --camera-height 720 --render-hz 10 --realtime
```

Terminal 2，检查传感器话题：

```bash
./check_hunter_isaac_topics.sh 45
./check_hunter_isaac_rates.sh
```

Terminal 3，启动 FAST-LIO：

```bash
./run_hunter_fastlio.sh
```

Terminal 4，键盘控制：

```bash
./run_hunter_isaac_keyboard.sh
```

发生建图漂移时运行：

```bash
./check_hunter_slam_drift.sh 20
```

这个诊断脚本会对比 FAST-LIO `/Odometry` 和 Isaac `/isaac/ground_truth/odometry`，同时输出 LiDAR/IMU 频率、点云字段、点数、最新控制量和相对 XY/z/yaw 漂移。

## FSGP-BGK / SEB 地形增强链路

当前分支新增 `fsgp_bgk_se2` ROS 2 集成包。它订阅现有 `terrain_analyzer` 的 `SE2Grid`，使用 GPU 优先、CPU 托底的 BGK 风格地形平滑和 SEB 航向扫描，发布增强后的 yaw-aware 风险地图：

```bash
./run_hunter_fsgp_bgk_se2.sh
```

让 `terrain_nav` 消费增强地图：

```bash
./run_hunter_fsgp_bgk_nav.sh
```

对应链路：

```text
FAST-LIO2 -> terrain_analyzer -> /car_0_terrain_analyzer/fused_map
          -> fsgp_bgk_se2 -> /car_0_fsgp_bgk/fused_map
          -> terrain_nav(risk_query_mode=se2_yaw)
```

## 重要说明

- Isaac Sim 中 `/Odometry` 留给 FAST-LIO，Isaac 真值里程计使用 `/isaac/ground_truth/odometry`，避免 RViz 和控制反馈混用两个里程计源。
- Isaac RTX PointCloud2 路径默认关闭伪造逐点时间：`preprocess.synthetic_point_time: false`。这可以避免没有真实 per-point timestamp 时 FAST-LIO 在转弯/坡面运动中过度去畸变。
- 当前 FAST-LIO-only 是主要建图验证链路；FAST-LIVO2 和 D455 配置保留，但多传感器同步、标定和部署稳定性还需要后续继续验证。
- 大型地图、模型、点云、扫描 CSV、图片和 PDF 资产通过 Git LFS 管理。
- 本仓库不提交 `build/`、`install/`、`log/`、本地虚拟环境和 rosbag 运行数据。

更多细节见：

- `docs/FSGP_BGK_INTEGRATION_PLAN.md`
- `src/isaac_sim_hunter_livo2/docs/ISAAC_SIM_LIVO2.md`
- `docs/FAST_LIVO2_D455_INTEGRATION.md`
- `NEOS_REAL_DEPLOYMENT_GUIDE.md`
