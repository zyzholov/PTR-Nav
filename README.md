# PTR-Nav

PTR-Nav 是一个面向复杂非结构化地形的 ROS 2 UGV 导航、建图、仿真与规划控制工程。当前分支记录 Isaac Sim、FAST-LIO2、GPU 地形分析、FSGP-BGK/SEB 增强与规划控制的集成开发状态，保留源码、配置、模型和脚本，不提交 `build/`、`install/`、`log/` 等本地编译/运行产物。

## 当前分支状态

本次版本分支为 `isaac-gpu-nav-baseline-20261010`，基于 `fsgp-bgk-se2-integration` 保存当前集成与实验基线，不修改仓库默认分支，也不代表已完成全场景验证。

首轮实验仅记录我们方法自身的数据，未运行外部方法对比：6 次流程门控测试中 3 次通过；静态绕障仍有失败，历史 `/Laser_map` 动态残影清除仍有不足，台阶测试出现控制延迟异常。实验协议与详细记录见 [动态地图与导航实验说明](docs/DYNAMIC_MAP_EXPERIMENTS.md)，模块启动与接口见 [Isaac 链路说明](docs/HUNTER_ISAAC_STACK.md) 和 [脚本目录说明](scripts/README.md)。

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

## Isaac Sim 模块化链路

当前仿真链路不再使用“大一键脚本”把所有节点绑在一起。按下面的顺序分别启动，每个终端只负责一个模块：

Terminal 1，仿真环境：

```bash
scripts/sim/start_simulation.sh
```

需要 D455 画面时：

```bash
scripts/sim/start_simulation.sh --camera-enabled
```

默认建图档位关闭相机以保持实时性；GUI 中启用后默认得到 960x540 RGB/深度图像。headless 建图档位会把相机分辨率限制到 640x360，并使用简化地表材质。需要更清晰的 GUI 图像时可传入 `--camera-width 1280 --camera-height 720 --render-quality quality`，但需要重新检查实时率。

Terminal 2，FAST-LIO2 建图：

```bash
scripts/sim/start_mapping.sh rviz:=true
```

Terminal 3，GPU 地形分析前端 + FSGP-BGK/SEB 增强：

```bash
scripts/sim/start_terrain_analysis.sh terrain_backend:=gpu fsgp_gpu_backend:=cupy
```

Terminal 4，规划与底盘控制：

```bash
scripts/sim/start_planning_control.sh
```

手动控制时不要启动 Terminal 4，改为：

```bash
scripts/sim/start_keyboard_control.sh
```

仅验证传感器和建图时，可以只启动前两个模块。无 GPU 时将 Terminal 3 改为：

```bash
scripts/sim/start_terrain_analysis.sh terrain_backend:=cpu fsgp_gpu_backend:=cpu
```

诊断脚本也已经归档到仿真目录：

```bash
scripts/sim/checks/topics.sh 45
scripts/sim/checks/rates.sh
scripts/sim/checks/slam_drift.sh 20
scripts/sim/checks/control_response.sh 0.8 0.30 6 steer
```

其中 `control_response.sh` 要在没有规划器发布 `/cmd_vel` 时运行。

FAST-LIO 点云话题含义：

- `/cloud_registered`：当前一帧在 `map` 坐标系下的注册点云，供实时地形分析使用。
- `/Laser_map`：经过体素去重的累计增量地图，当前 Isaac 配置以 1 Hz 发布，RViz 默认显示此话题；首次初始化后需要等到下一帧完成匹配。
- `/livox/lidar`：仿真雷达原始实时点云，RViz 默认关闭，排查传感器时可手动打开。

## FSGP-BGK / SEB 地形增强链路

当前分支新增 `fsgp_bgk_se2` ROS 2 集成包。它订阅现有 `terrain_analyzer` 的 `SE2Grid`，使用 CPU/CuPy 自适应后端的 BGK 风格地形平滑和 SEB 航向扫描，发布增强后的 yaw-aware 风险地图。常规使用一个地形分析入口启动这两个计算节点：

```bash
scripts/sim/start_terrain_analysis.sh
```

默认 `fsgp_gpu_backend:=auto` 会按计算规模选择后端：当前 `104x104x36` 地图会走 CuPy，仅很小的地图保留 CPU 以避开 kernel 启动开销。强制 GPU 写法是：

```bash
scripts/sim/start_terrain_analysis.sh terrain_backend:=gpu fsgp_gpu_backend:=cupy
```

只需基础地形图时设 `enhance:=false`。若确实需要在两个终端分别调试前端与增强节点，再运行 `scripts/sim/optional/start_fsgp_bgk.sh gpu_backend:=cupy`。

自主导航模式，让 `terrain_nav` 消费增强地图并独占 `/cmd_vel`：

```bash
scripts/sim/start_planning_control.sh
```

手动驾驶并同时观察 GPU 地形分析/FSGP 增强地图时，必须关闭导航控制输出：

```bash
scripts/sim/start_terrain_analysis.sh
scripts/sim/start_keyboard_control.sh
```

不要同时运行键盘和 `terrain_nav`。键盘脚本会检测已有 `/cmd_vel` 发布者并拒绝启动，避免两个控制器互相覆盖。

对应链路：

```text
FAST-LIO2 -> terrain_analyzer -> /car_0_terrain_analyzer/fused_map
          -> fsgp_bgk_se2 -> /car_0_fsgp_bgk/fused_map
          -> terrain_nav(risk_query_mode=se2_yaw)
```

这里的 `terrain_analyzer` 是地形感知前端：它把当前帧 `/cloud_registered`
和 `/Odometry` 转成局部高程、方差、SDF、障碍和已有航向切片的基础风险
`SE2Grid`。它不做高斯回归，也不发布控制指令。`fsgp_bgk_se2` 是增强后端：
它对前端栅格执行 BGK 风格加权平滑，并用 SEB 按车辆航向扫描 footprint，生成
`/car_0_fsgp_bgk/fused_map` 中的航向相关风险层。最后 `terrain_nav` 只消费
增强地图并发布 `/cmd_vel`。因此三个计算阶段分开是为了可观测、可替换和避免控制器
抢占话题，不是重复计算同一件事。当前接入版尚未移植上游完整的稀疏高斯过程
(SGP)，不要把 BGK 风格平滑误认为完整 FSGP 算法。

旧的一键组合入口和旧 Gazebo/实验脚本已移到 `scripts/old/`，不再作为当前
工程入口。实机入口位于 `scripts/real/`；仿真入口位于 `scripts/sim/`。

## 重要说明

- Isaac Sim 中 `/Odometry` 留给 FAST-LIO，Isaac 真值里程计使用 `/isaac/ground_truth/odometry`，避免 RViz 和控制反馈混用两个里程计源。
- Isaac RTX PointCloud2 路径默认关闭伪造逐点时间：`preprocess.synthetic_point_time: false`。这可以避免没有真实 per-point timestamp 时 FAST-LIO 在转弯/坡面运动中过度去畸变。
- 当前 FAST-LIO-only 是主要建图验证链路；FAST-LIVO2 和 D455 配置保留，但多传感器同步、标定和部署稳定性还需要后续继续验证。
- 大型地图、模型、点云、扫描 CSV、图片和 PDF 资产通过 Git LFS 管理。
- 本仓库不提交 `build/`、`install/`、`log/`、本地虚拟环境和 rosbag 运行数据。

更多细节见：

- `docs/HUNTER_ISAAC_STACK.md`
- `docs/FSGP_BGK_INTEGRATION_PLAN.md`
- `src/isaac_sim_hunter_livo2/docs/ISAAC_SIM_LIVO2.md`
- `docs/FAST_LIVO2_D455_INTEGRATION.md`
- `NEOS_REAL_DEPLOYMENT_GUIDE.md`
