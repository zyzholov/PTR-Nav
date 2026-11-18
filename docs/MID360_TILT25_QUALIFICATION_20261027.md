# MID360 倾斜 25°：修正与定向资格

本页记录本地源码、真实 RTX 射线和物理导航检查。未运行正式批量实验，未推送 GitHub。此前水平 Ouster 配置的资格不自动转移到本版本；RoboSense RS-LiDAR-32 尚未接入。

## 修正内容

1. 默认 `--lidar-quality mid360_csv`：每帧 20,000 条发射光束、10 帧/秒，推进完整 CSV 的 40 个阶段，而非每帧重复第一段。四秒后循环；这是有限扫描表近似，不是硬件光学机制、真实 Livox CustomMsg 或逐点运动畸变的等价复现。有效回波数取决于场景；发射光束不能全算成回波。
2. 原始安装恢复 +25°，同步实际 Isaac FAST-LIO 配置及可选 FAST-LIVO 相机/IMU 外参。OmniLidar 直接定义在该安装节点下，避免 Replicator 将嵌套路径改名。Gazebo 配置未改。
3. 修正新配置引入的严重畸变：Isaac 5.1 的扫描缓冲节点按球坐标 GMO 输入再转换成 XYZ；此前显式设置笛卡尔 GMO，导致二次错误解释。现在内部为 `SPHERICAL`，ROS 输出仍为笛卡尔米制 XYZ。
4. 保留 `point_filter_num=1`、注册体素 0.20 米、内部地图体素 0.25 米、累计显示体素 0.10 米。增加真实角度覆盖，不插值伪造累计点。
5. 实时着色层保留一帧原始完整注册回波，RViz `Decay Time=0`，避免两帧重叠和到期闪烁；发布器在真实过期后清空一次，不重写源时间。颜色表示距车辆的水平距离；圆状等色线不代表扫描几何或反射强度。

MID360 +25° 的后方地面盲区保留。不能为了画满地图而把雷达改水平或伪造后方观测。

## 起步与补全的模型边界

默认仍为起始未知。显式仿真配置可打开 `stable_stance` 与 `startup_support_directional`：车辆先稳定，要求至少 24 个周边实测地面格、足够二维展开、地面平面残差、车体离地高度及姿态一致。通过后只冻结初始足迹，真实地面矛盾或占据会撤销；它不是实测地面，实车禁用。

显式共享 GP 允许距实测支撑不超过 0.75 米的外推；通常仍有 4 厘米绝对标准差门限。仅已获授权且与实测平面一致的固定起始足迹，允许保留最高 8 厘米的真实绝对标准差，再用同一后验的共享协方差计算坡度/台阶事件。没有缩小该方差、加入假设点训练或改变 5% 状态、10% 局部路径、5% 制动预算。较大的公共高度偏差可在几何差分中抵消；不能将其拆成独立误差后误拒整条起步路径。

这是条件高斯模型与显式仿真假设，不是经验概率校准或实车安全认证。未支持区域保留未知。

## 实际记录

各目录保存源码快照、配置/二进制哈希、原始轨迹和时间戳。到达、显示、时效与残影清除分别验收，不混算。

| 检查 | 结果与边界 | 本地数据 |
|---|---|---|
| 水平、+25°、−25° 已知平面射线 | 三种姿态各 10 帧；最坏帧世界高程误差 P95 ≤4.2 毫米，+25° 时 ≤3.6 毫米 | `log/mid360_rtx_geometry_qualified_20261027.json` |
| 实际回波角度推进 | 8 秒 80 帧，约 10 Hz；0.25° 角度格并集为首帧约 18.1 倍；累计地图约 6.45 万点，随后超过 11 万点 | `log/modular_smoke_20261027_145248/mid360_scan.json` |
| Mars GUI 2 米目标 | 整链通过，停稳真值误差 0.099 米，首次控制响应 0.040 秒，控制约 50 Hz | `log/modular_smoke_20261027_145248` |
| 突现后持续占道，9 米目标 | 整链通过；22.31 秒到达，真值误差 0.102 米，侧向绕行最大 1.92 米，最小采样车身净距 0.820 米；障碍未离开 | `log/modular_smoke_20261027_150056` |
| 障碍离开后的残影清除 | **未通过总验收**。车辆到达，风险图 0.28 秒恢复，实时占据 0.08 秒恢复；累计图约 10 秒后仍残留 416 个旧障碍点 | `log/modular_smoke_20261027_150454` |
| 高程模式显示晚加入 | 一次显示资格在发目标前失败；增加接收计数/时间戳诊断。后续同模式通过，但不能据此声称间歇问题已根除 | `log/modular_smoke_20261027_145603`、`150056` |

短程运动窗口：雷达/注册扫描/里程计约 10 Hz、外置 IMU 约 199 Hz、风险图约 8 Hz，风险图接收时最大源年龄 0.275 秒。不是硬实时或连续碰撞证明。终点航向、坡道及快速横穿未由上述两个 MID360 到达记录完成验收。

全套当前 FSGP 与仿真检查 Python 回归：561 项通过，见 `log/mid360_full_regression_20261027.log`。GPU 数值检查允许不同 float32 归约顺序产生不超过约 3e-6 的差异，并额外检查 5%/10% 判定一致。

无命中射线当前**没有接入生产清除链路**。诊断读取完整 RTX GMO 的发射数组/有效标记；还需验证物体遮挡、无反射材质、帧归属和源时间关联，不能从话题缺帧或丢包推断自由空间。累计图残影问题保持打开。

## 四个终端顺序启动

每个终端进入工程目录，沿用主 README 的同一 ROS domain/Fast DDS 环境。不要在这组脚本旁再启动旧仿真或旧建图节点。

```bash
# 终端 1：GUI 仿真，默认 MID360 +25°
scripts/sim/start_simulation.sh --lidar-quality mid360_csv

# 终端 2：FAST-LIO2，共用后面的 RViz
scripts/sim/start_mapping.sh rviz:=false

# 终端 3：GPU 地形分析、补全与共享风险、高程/点云 RViz
scripts/sim/start_terrain_analysis.sh terrain_backend:=gpu fsgp_gpu_backend:=cupy \
  prediction_backend:=feature_sgp gp_device:=cuda \
  probability_model:=shared_gp gp_complete_supported:=true \
  probability_gp_max_distance_m:=1.25 probability_gp_max_extrapolation_m:=0.75 \
  startup_support_mode:=stable_stance startup_support_directional:=true \
  obstacle_memory_topic:=/Laser_map obstacle_memory_role:=advisory

# 终端 4：规划与控制
scripts/sim/start_planning_control.sh chance_allow_gp_completion:=true \
  chance_allow_joint_model:=true chance_allow_startup_prior:=true reverse_recovery_enable:=true
```

手动驾驶时，终端 4 改为 `scripts/sim/start_keyboard_control.sh`，不要与自动控制同时发命令。当前跟踪器仍是混合 Pure Pursuit/Stanley，不是 MPPI。
