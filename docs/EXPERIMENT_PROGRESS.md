# 实验进度说明文档

## 一、项目概述

### 1.1 研究目标

实现基于激光雷达的无人地面车辆 (UGV) 在非结构化地形上的自主导航。系统需要:

1. **实时感知**: 使用激光雷达 + IMU 构建环境的三维地图
2. **地形分析**: 从点云中提取地面高程、障碍物、可通行性信息
3. **路径规划**: 在地形约束下规划出安全、可行的轨迹
4. **轨迹跟踪**: 通过模型预测控制 (MPC) 精确跟踪规划轨迹

### 1.2 硬件平台

| 组件 | 型号 | 作用 |
|------|------|------|
| 车辆底盘 | Hunter SE | Ackermann 转向模型，轮距 0.55m |
| 激光雷达 | Livox MID360 | 非重复扫描式，发布 `/livox/lidar` |
| IMU | 外置 IMU 模块 | Isaac 主话题 `/imu/data_raw`，Gazebo/旧配置兼容 `/livox/imu` |

### 1.3 软件栈

| 层级 | 模块 | 作用 |
|------|------|------|
| 仿真 | Gazebo Classic | 物理仿真环境 |
| 里程计 | FAST_LIO2 | 激光雷达惯性里程计，输出 6DoF 位姿 |
| 建图 | Terrain Analyzer | 构建高程图、风险图、SDF 图 |
| 规划 | ARC-Planner | Kino A* 前端 + MINCO 轨迹优化后端 |
| 控制 | MPC Controller | CasADi 非线性模型预测控制 |
| 可视化 | RViz2 | 地图、轨迹、车辆状态可视化 |

---

## 2026-10-08 可视化与链路回归

本轮将地形分析与规划共用的 RViz2 配置独立为
`src/fsgp_bgk_se2/rviz/hunter_terrain_navigation.rviz`。默认视图包含地形高程、跟随车辆航向的 FSGP-BGK/SEB 风险、实时障碍、规划路径、目标、FAST-LIO 里程计和车辆模型；累计 `/Laser_map` 保留为关闭的诊断层，以免启动时遮挡局部导航图并拖慢渲染。

`se2_grid_converter` 已修正矩形网格、环形缓存起始索引、行/列存储、数据偏移和缺失图层静默回退问题。风险点云从 `bgk_mean` 取高程贴地，并按 `/Odometry` 选择 SO(2) 航向切片。新增 `visualization_probe.py` 会验证两张点云的非空性、坐标范围、共同网格和贴地高度差。

实测结果：

* `se2_grid_converter` 与 `fsgp_bgk_se2` 共 20 个测试通过。
* GPU 无 GUI 全链路 `log/modular_smoke_20261008_095419` 通过；高程/风险点云分别 4254 点，共同网格 4254 点，中位高度差 0.020 m；LiDAR 10.00 Hz、外置 IMU 199.5 Hz、地形分析 10.03 Hz、FSGP-BGK 5.00 Hz。
* GUI 全链路 `log/modular_smoke_20261008_095905` 通过，截图为 `log/modular_smoke_20261008_095905/rviz_snapshot.png`。
* 无障碍 2 m 目标 `log/modular_smoke_20261008_101011` 通过，车辆运动 1.48 m，最终目标误差 0.52 m，FAST-LIO 与 Isaac 真值的最大 XY 误差 0.003 m。
* detour 导航回归 `log/modular_smoke_20261008_095618` 未通过：规划器已经输出约 -0.40 rad 转向，物理仿真也测到约 -0.52 rad/s 横摆，但车辆在实时障碍附近停止。该问题属于障碍/规划闭环，不能以 RViz 显示通过代替，后续需要单独修复。

本轮没有提交或推送 GitHub。

---

## 二、系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Gazebo 仿真环境                           │
│  Hunter SE + Livox MID360 + IMU                                 │
└──────────────┬──────────────────────┬───────────────────────────┘
	               │ cmd_vel              │ /livox/lidar + 外置 IMU
               ▼                      ▼
┌──────────────────────┐  ┌───────────────────────────────────────┐
│  Gazebo 车辆插件      │  │  FAST_LIO2                            │
│  Ackermann 驱动       │  │  iEKF + ikd-Tree                     │
└──────────────────────┘  └──────────┬────────────────────────────┘
                                     │
                    /Odometry        │ /cloud_registered
                    (6DoF 位姿)      │ (世界坐标系点云)
                                     ▼
                          ┌──────────────────────┐
                          │  Terrain Analyzer     │
                          │  高程图 + 风险图 + SDF │
                          └──────────┬───────────┘
                                     │
                    fused_map        │ sdf_map
                    (高程+风险)       │ (距离场)
                                     ▼
                          ┌──────────────────────┐
                          │  ARC-Planner          │
                          │  Kino A* + MINCO 优化  │
                          └──────────┬───────────┘
                                     │
                                     │ /planner/trajectory
                                     ▼
                          ┌──────────────────────┐
                          │  MPC Controller       │
                          │  CasADi 非线性 MPC     │
                          └──────────┬───────────┘
                                     │
                                     │ /hunter_se/cmd_vel
                                     ▼
                          ┌──────────────────────┐
                          │  Gazebo 车辆执行       │
                          └──────────────────────┘
```

---

## 三、已完成的工作

### 3.1 基础仿真环境搭建

- [x] Gazebo Classic + Hunter SE 车辆模型配置
- [x] Livox MID360 激光雷达插件集成
- [x] IMU 插件配置
- [x] 多地图支持 (mars_gale 等)
- [x] 启动脚本 (`scripts/old/sim/run_hunter_gazebo.sh`, `scripts/sim/start_mapping.sh`)

### 3.2 FAST_LIO2 集成

- [x] 点云预处理与 IMU 融合
- [x] 6DoF 位姿估计 (`/Odometry`, ~200Hz)
- [x] 世界坐标系点云配准 (`/cloud_registered`)
- [x] 局部点云地图 (`/cloud_local_map`)

### 3.3 地形分析器 (Terrain Analyzer)

- [x] 高程图构建: 非对称 EMA，低点优先更新
- [x] 风险图构建: 坡度 + 曲率 + 台阶高度加权合成
- [x] SDF 图构建: Chamfer 3×3 两遍距离传播
- [x] 垂直障碍物检测: 碰撞带判断 + 生命周期管理
- [x] Ego Body Filter: 过滤车身内的激光点
- [x] 障碍物膨胀: 安全裕度扩展
- [x] Ego Clear: 清除车辆本体范围内的障碍物标记
- [x] 滑动窗口: 地图随车辆滚动更新
- [x] 可视化发布: 高程/SDF/风险/障碍物点云

### 3.4 路径规划器 (ARC-Planner)

- [x] Kino A* 前端搜索: 自行车模型运动学扩展
- [x] MINCO 轨迹表示: 5 阶多项式，弧长参数化
- [x] ALM + L-BFGS 后端优化: 约束优化求解
- [x] 动力学约束: 曲率、速度、纵向/横向加速度
- [x] 地形感知: 坡度修正速度/加速度极限
- [x] 障碍物避让: ESDF 代价 + 车辆 footprint 检测
- [x] 多段换挡: 支持前进/后退切换
- [x] 安全验证: 优化后检查 SDF 和风险阈值
- [x] 重规划 FSM: 轨迹过期、跟踪误差、碰撞触发

### 3.5 MPC 控制器

- [x] CasADi 非线性 MPC 求解
- [x] 自行车模型状态转移
- [x] 轨迹跟踪: `/reference_path` 滑动更新
- [x] 预测路径可视化: `/predict_path`
- [x] 车辆历史轨迹可视化
- [x] 速度 + 转向角控制输出

### 3.6 文档编写

- [x] README.md: 项目总体说明
- [x] TERRAIN_MAPPING_DIAGRAM.md: 地形建图架构框图
- [x] TERRAIN_MAPPING_DETAILED.md: 地形建图详细流程与原理
- [x] PLANNING_DETAILED.md: 路径规划详细流程与原理
- [x] GAZEBO_FASTLIO_TERRAIN_PLANNING_NEXT_CONTEXT.md: 上下文交接文档
- [x] GAZEBO_FASTLIO_TERRAIN_PLANNING_DEBUG_GUIDE.md: 调试指南
- [x] NEOS_REAL_DEPLOYMENT_GUIDE.md: 实车部署指南

---

## 四、已发现并修复的问题

### 4.1 地形分析器: Ego Body Filter 盲区 (已修复)

**问题描述:**
Ego Body Filter 的过滤区域过大 (前方 1.10m, 后方 0.85m, 半宽 0.68m)，导致车身附近约 1.1m 范围内的障碍物无法被检测到。这使得:
- SDF 图在车辆附近缺失障碍物距离信息
- 风险图在车辆附近缺失障碍物风险区域
- 规划器可能误判车辆附近为可通行区域

**修复方案:**
缩小 Ego Body Filter 尺寸:
```yaml
# 修改前
ego_body_filter_front_m: 1.10
ego_body_filter_back_m: 0.85
ego_body_filter_half_width_m: 0.68
ego_clear_radius: 0.62

# 修改后
ego_body_filter_front_m: 0.50
ego_body_filter_back_m: 0.50
ego_body_filter_half_width_m: 0.30
ego_clear_radius: 0.30
```

**修复效果:**
- 检测盲区从 ~1.1m 缩小到 ~0.5m
- 车辆附近的障碍物能够被正确检测并显示 SDF 距离和风险区域

**修改文件:**
- `src/uneven_mapping/terrain_analyzer/params/hunter_fastlio_registered_live.yaml`

### 4.2 MPC: smooth_yaw 死循环 (已修复)

**问题描述:**
`smooth_yaw` 函数中的 `while` 循环在倒车场景下可能死循环。当参考轨迹的航向与车辆航向相差约 π (倒车方向) 时:
```
dyaw = ref_points[0].theta - now_state.theta  // ≈ ±π
while (dyaw >= M_PI / 2):   // 条件满足
    ref_points[0].theta -= M_PI * 2
    dyaw = ...               // 仍然 ≈ ±π，无限循环
```

**修复方案:**
在进入平滑逻辑前，先将航向归一化到 [-π, π]:
```cpp
double dyaw = normalize_angle(ref_points[0].theta - now_state.theta);
```

**修改文件:**
- `src/mpc_controller/include/mpc/mpc.h`

### 4.3 MPC: 航向约定不匹配 (已修复)

**问题描述:**
DP 轨迹的 `getYaw()` 返回运动方向 (倒车时为反方向)，但 MPC 期望的是车身朝向。这导致倒车时 MPC 使用错误的航向进行跟踪。

**修复方案:**
在倒车段将 theta 翻转 π:
```cpp
if (eta < 0) {  // 倒车段
    theta = normalize_angle(theta + M_PI);
}
```

**修改文件:**
- `src/mpc_controller/include/utils/traj_anal.hpp`

### 4.4 MPC: 曲率符号反转 (已修复)

**问题描述:**
`getCur()` 返回的曲率乘以了 eta (方向标志)，导致倒车时曲率符号反转，转向方向错误。

**修复方案:**
添加 `getRawCur()` 方法返回不带符号翻转的原始曲率。

**修改文件:**
- `src/mpc_controller/include/utils/dptraj.hpp`
- `src/mpc_controller/include/utils/minco_traj.hpp`

---

## 五、已发现但回退的修改

### 5.1 规划器: A* 前端 Footprint 碰撞检测 (已回退)

**问题描述:**
Kino A* 前端只检查路径中心点的 SDF 距离，不检查车辆 footprint 的四个角点。这导致规划器可能生成过于贴近障碍物的路径。

**尝试的修复:**
添加 9 点 footprint 采样碰撞检测。

**回退原因:**
修改后规划器变得几乎不可用，用户要求回退所有规划器和 MPC 的修改。

**当前状态:**
保留原始的单点碰撞检测，通过后端优化的 ESDF 代价补偿。

### 5.2 规划器: 障碍物距离代价 (已回退)

**问题描述:**
A* 前端的代价函数中没有障碍物距离项，导致规划器不主动远离障碍物。

**尝试的修复:**
添加基于 SDF 的 clearance penalty。

**回退原因:**
同上，用户要求回退。

**当前状态:**
依赖后端优化的 ESDF 代价进行避障。

### 5.3 MPC: 倒车航向修复系列 (已回退)

**问题描述:**
倒车时 MPC 跟踪误差大，原因包括:
- smooth_yaw 死循环
- 航向约定不匹配
- 曲率符号反转

**尝试的修复:**
修改 smooth_yaw、添加 theta 翻转、添加 getRawCur。

**回退原因:**
同上，用户要求回退。后续用户自行修改了部分代码。

**当前状态:**
用户已独立修改 `mpc.h`，可能已解决部分问题。

---

## 六、当前状态与待解决问题

### 6.1 当前系统状态

| 模块 | 状态 | 说明 |
|------|------|------|
| Gazebo 仿真 | ✅ 正常 | 多地图支持，车辆物理模型正常 |
| FAST_LIO2 | ✅ 正常 | 位姿估计准确，点云质量良好 |
| Terrain Analyzer | ✅ 已修复 | Ego Body Filter 盲区已缩小 |
| ARC-Planner | ⚠️ 有已知问题 | 前端碰撞检测粗糙，有时绕远 |
| MPC Controller | ⚠️ 用户自行修改中 | 倒车跟踪待验证 |
| RViz 可视化 | ✅ 正常 | 地图/轨迹/车辆状态正常显示 |

### 6.2 待解决问题

#### 问题 1: 规划器有时绕远或绕圈

**现象:** 规划器生成的路径不是最优的，有时会绕远路或出现很弯的局部轨迹。

**可能原因:**
- Kino A* 前端的启发函数只考虑欧氏距离，不考虑航向
- 代价函数中没有障碍物距离项，不主动远离障碍物
- 时间预算过短 (0.40s)，搜索不充分

**建议方向:**
- 改进启发函数，加入航向一致性
- 在后端优化中增强 ESDF 权重
- 适当增加搜索时间预算

#### 问题 2: 倒车时横向转向跟踪误差

**现象:** 倒车时可以正常倒车，但拐弯时跟踪误差较大。

**可能原因:**
- MPC 的 smooth_yaw 在倒车时可能仍有问题
- 曲率符号在倒车段需要翻转
- 航向约定 (运动方向 vs 车身朝向) 不一致

**当前进展:**
用户已独立修改 `mpc.h`，可能已解决部分问题，需实车测试验证。

#### 问题 3: 靠近障碍物时 SDF/风险层异常

**现象:** 车辆靠近障碍物时，terrain analyzer 的风险/SDF 可视化曾出现被清空的现象。

**已采取措施:**
- 缩小 Ego Body Filter 盲区
- 添加 Ego Clear 的 0.12m 保留环

**当前状态:**
已修复，需持续观察。

#### 问题 4: /reference_path 更新时机

**现象:** 参考路径需要伴随车辆滑动更新，不能等车辆超过末端才更新。

**当前状态:**
MPC 中已实现滑动参考点 (`getFutureWps`)，但更新时机可能需要优化。

---

## 七、关键配置参数

### 7.1 地形分析器参数

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `resolution` | 0.2m | 栅格分辨率 |
| `grid_size_x/y` | 104 | 栅格地图尺寸 |
| `obstacle_min_height` | 0.28m | 障碍物最小高度 |
| `obstacle_max_height` | 1.45m | 障碍物最大高度 |
| `sdf_obstacle_slope` | 1.10 rad | 硬障碍坡度阈值 (~63°) |
| `step_height_soft` | 0.22m | 软台阶高度阈值 |
| `risk_weight_slope` | 0.22 | 坡度风险权重 |
| `risk_weight_curvature` | 0.33 | 曲率风险权重 |
| `risk_weight_step_height` | 0.45 | 台阶高度风险权重 |
| `ego_body_filter_front_m` | 0.50m | 本体过滤器前方距离 |
| `ego_body_filter_half_width_m` | 0.30m | 本体过滤器半宽 |
| `vertical_obstacle_inflation_m` | 0.41m | 障碍物膨胀半径 |

### 7.2 规划器参数

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `vmax` | 0.70 m/s | 最大速度 |
| `kmax` | 0.88 | 最大曲率 |
| `safeMargin` | 0.42m | 安全裕度 |
| `esdfWei` | 1.05 | ESDF 代价权重 |
| `front_end/traj_back_penalty` | 2.0 | 后退代价倍数 |
| `front_end/traj_gear_switch_penalty` | 14.0 | 换挡代价 |
| `fsm/time_budget` | 0.30s | 规划时间预算 |

### 7.3 MPC 参数

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `dt` | 0.2s | 控制周期 |
| `Npre` | 5 | 预测步数 |
| `max_steer` | π/4 | 最大转向角 |
| `max_speed` | 55/3.6 m/s | 最大速度 |
| `wheel_base` | 0.5m | 轴距 |

---

## 八、实验场景

### 8.1 测试地图

| 地图 | 特点 | 测试重点 |
|------|------|----------|
| `mars_gale` | 复杂地形，有坡度、台阶、障碍物 | 全链路测试 |
| 默认地图 | 简单平面 | 基础功能验证 |

### 8.2 测试用例

| 用例 | 测试内容 | 状态 |
|------|----------|------|
| 前进导航 | 从 A 点到 B 点的前进路径规划与跟踪 | ✅ 基本可用 |
| 倒车导航 | 需要倒车才能到达的目标点 | ⚠️ 待验证 |
| 避障导航 | 绕过静态障碍物到达目标点 | ⚠️ 有时绕远 |
| 坡度地形 | 在有坡度的地形上行驶 | ✅ 基本可用 |
| 台阶地形 | 跨越小台阶 | ✅ 基本可用 |
| 近距离障碍物 | 在障碍物附近规划和行驶 | ✅ 已修复 |

---

## 九、启动命令速查

### 9.1 完整链路测试

```bash
# 终端 1: Gazebo + 车辆
cd /home/z/workspace_zyz/seb_clean
scripts/old/sim/run_hunter_gazebo.sh map:=mars_gale

# 终端 2: FAST_LIO2
cd /home/z/workspace_zyz/seb_clean
scripts/sim/start_mapping.sh

# 终端 3: 地形分析 + 规划 + MPC + RViz
cd /home/z/workspace_zyz/seb_clean
source install/setup.bash
ros2 launch plan_manage hunter_gazebo_fastlio_nav.launch.py terrain:=true planner:=true mpc:=true rviz:=true
```

### 9.2 分层测试

```bash
# 只测地形分析
ros2 launch plan_manage hunter_gazebo_fastlio_nav.launch.py terrain:=true planner:=false mpc:=false rviz:=true

# 只测规划器
ros2 launch plan_manage hunter_gazebo_fastlio_nav.launch.py terrain:=true planner:=true mpc:=false rviz:=true

# 测试规划 + MPC
ros2 launch plan_manage hunter_gazebo_fastlio_nav.launch.py terrain:=true planner:=true mpc:=true rviz:=true
```

### 9.3 局部编译

```bash
cd /home/z/workspace_zyz/seb_clean
colcon build --packages-select terrain_analyzer path_search plan_manage mpc_controller --symlink-install
source install/setup.bash
```

### 9.4 必查话题

```bash
# 传感器频率
ros2 topic hz /livox/lidar
ros2 topic hz /Odometry

# 地形产品
ros2 topic hz /car_0_terrain_analyzer/fused_map
ros2 topic hz /car_0_terrain_analyzer/sdf_map

# 规划输出
ros2 topic hz /planner/trajectory
ros2 topic hz /planner/dptrajectory

# 控制输出
ros2 topic echo /hunter_se/cmd_vel
```

---

## 十、参考文档索引

| 文档 | 内容 |
|------|------|
| [README.md](../README.md) | 项目总体说明 |
| [TERRAIN_MAPPING_DIAGRAM.md](TERRAIN_MAPPING_DIAGRAM.md) | 地形建图架构框图 |
| [TERRAIN_MAPPING_DETAILED.md](TERRAIN_MAPPING_DETAILED.md) | 地形建图详细流程与原理 |
| [PLANNING_DETAILED.md](PLANNING_DETAILED.md) | 路径规划详细流程与原理 |
| [MPC_CONTROL_DETAILED.md](MPC_CONTROL_DETAILED.md) | 轨迹跟踪控制 (MPC) 详细流程与原理 |
| [PROJECT_SUMMARY_AND_FUTURE.md](PROJECT_SUMMARY_AND_FUTURE.md) | 工程总结与未来工作计划 |
| [EXPERIMENT_PROGRESS.md](EXPERIMENT_PROGRESS.md) | 实验进度说明 (本文档) |
| [GAZEBO_FASTLIO_TERRAIN_PLANNING_NEXT_CONTEXT.md](../GAZEBO_FASTLIO_TERRAIN_PLANNING_NEXT_CONTEXT.md) | 上下文交接文档 |
| [GAZEBO_FASTLIO_TERRAIN_PLANNING_DEBUG_GUIDE.md](../GAZEBO_FASTLIO_TERRAIN_PLANNING_DEBUG_GUIDE.md) | 调试指南 |
| [NEOS_REAL_DEPLOYMENT_GUIDE.md](../NEOS_REAL_DEPLOYMENT_GUIDE.md) | 实车部署指南 |

## 2026-10-03 继续回归：参数链路与动态地图

本轮先修复了 FAST-LIO 动态地图参数声明不完整的问题。此前 YAML 中的
`dynamic_ground_clear_frames`、`dynamic_ground_min_height` 和
`dynamic_expand_min_seeds` 没有全部显式声明，节点可能静默使用 C++ 默认值。
现在节点声明、读取并打印完整动态地图策略；`mapping_parameters.py` 会在启动后
通过 ROS 参数服务逐项核对配置。GPU 全链路的 68 项参数审计通过。

当前 Isaac 默认策略保持保守：直接自由射线清除开启，连续 3 帧证据；地面替换和
邻域扩张关闭。后两者属于实验性累计图历史清除，之前的多视角护栏试验出现过静态
障碍最低保留数下降，因此不能作为默认的实时清残影方案。

回归结果：

| 测试 | 结果 | 证据 |
|------|------|------|
| GPU 四模块静止链路 | 通过；LiDAR 10.02 Hz、IMU 200.04 Hz、地形 10.03 Hz、FSGP 5.01 Hz，FAST-LIO XY 最大误差 0.002 m | `log/modular_smoke_20261003_232904` |
| CPU fallback 静止链路 | 通过；LiDAR 10.02 Hz、IMU 200.01 Hz、地形新鲜时间戳 10.04 Hz、FSGP 7.48 Hz，FAST-LIO XY 最大误差 0.003 m | `log/modular_smoke_20261003_233944` |
| 动态护栏 | 通过；静态护栏最低 4/5，累计图观测清除 0.74 s，注册层障碍降为 0 | `log/modular_smoke_20261003_233119` |
| 静态障碍 detour | 通过；到达目标，最近障碍距离 1.473 m，风险图周期 P95 0.222 s、年龄 P95 0.09 s | `log/modular_smoke_20261003_234151` |
| 移动障碍绕行/清除 | 到达目标；风险/live 层 0.09/0.19 s 清除，但严格累计图 ROI 基线不足，按测试规则拒绝通过 | `log/modular_smoke_20261003_233427` |

因此目前可以确认：实时注册点云、风险图和 live obstacle 层可用于规划控制；
`/Laser_map` 仍是可视化累计历史，不能承诺在所有视角和遮挡条件下固定时间清除
动态残影。后续应补充有明确高障碍基线的移动障碍 fixture，再评估累计图清除，而
不是放宽静态安全判定。
