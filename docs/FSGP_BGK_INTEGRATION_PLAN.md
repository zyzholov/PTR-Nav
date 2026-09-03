# FSGP-BGK 与 SEB 集成设计说明

## 1. 结论

FSGP-BGK 可以嵌入当前工程，而且它和当前工程里的 `SE2Grid`/地形导航框架是匹配的。但它不应该直接替换 FAST-LIO2，也不建议直接把原版 `node_ros2.py` 接进主链路就用于规划。更稳的方式是：

1. FAST-LIO2 继续负责定位建图，输出 `/cloud_registered` 与 `/Odometry`。
2. 现有 `terrain_analyzer` 继续负责基础高程、方差、垂直障碍物和基础风险层。
3. 新增一个 FSGP-BGK/SEB 增强节点，同时消费 FAST-LIO2 的点云/里程计和 `terrain_analyzer` 的基础 `fused_map`，输出带航向维度的可通行性 `SE2Grid`。
4. 修改 `terrain_nav`，让规划器在需要时按照 `(x, y, yaw)` 查询风险，而不是把风险提前压平成二维 costmap。

也就是说，FSGP-BGK 更适合作为“地形可通行性增强层”，不是 SLAM 前端。它能提升“这个位置是否能走”的判断质量；SLAM 漂移仍主要受雷达点云质量、IMU 时间戳、外参、车体运动真实性影响。

## 2. 现有工程主链路

当前工程的核心链路可以理解为：

```text
Isaac/Gazebo Hunter + MID360/IMU/D455
        |
        | /livox/lidar
        | /livox/imu
        v
FAST-LIO2
        |
        | /cloud_registered
        | /Odometry
        v
terrain_analyzer
        |
        | /car_0_terrain_analyzer/fused_map
        |   layers: elevation, var, risk, valid, zbx, zby
        v
terrain_nav
        |
        | /hunter_se/cmd_vel
        v
Hunter chassis simulation/control
```

这里有一个关键限制：虽然 `fused_map` 类型是 `SE2Grid`，但当前 ROS2 版 `terrain_analyzer` 在发布时把同一个二维 `risk(x,y)` 写入所有 yaw bin。`terrain_nav` 又在 `rebuildCostmapLocked()` 中读取 `risk(x,y,0)` 并生成二维 costmap。因此当前主链路并没有真正利用“同一位置、不同航向、不同可通行性”。

这正是 FSGP-BGK + SEB 值得嵌入的地方。

## 3. FSGP-BGK 包现状

本地新增包路径：

```text
src/FSGP-BGK-ROS2
```

上游来源为：

```text
https://github.com/MarineRock10/FSGP-BGK-ROS2.git
```

该仓库声明为 MIT License。当前本地目录里还保留了它自己的 `.git`，如果后续要纳入当前 PTR-Nav 仓库，需要二选一处理：

- 作为 git submodule 管理。
- 或移除嵌套 `.git`，按 vendor 源码纳入当前仓库。

核心文件职责如下：

| 文件 | 作用 |
| --- | --- |
| `src/fsgp_bgk/python/f_sgp_bgk.py` | 主算法类 `TraversabilityAnalyzer`，完成局部点云下采样、特征选择、SGP/BGK 推理和可通行性计算 |
| `src/fsgp_bgk/python/node_ros2.py` | ROS2 Python 节点，订阅点云和里程计，发布 traversability 点云 |
| `src/fsgp_bgk/python/traversability_lib/sgp_model.py` | 基于 `gpytorch` 的 SGP 模型，使用 `InducingPointKernel` 与 `RQKernel` |
| `src/fsgp_bgk/python/traversability_lib/choose_point.py` | 使用 `torch` 做 GPU KNN、曲率/梯度特征提取，同时混用 Open3D/sklearn 做下采样和近邻 |
| `src/fsgp_bgk/python/traversability_lib/traversability.py` | 使用 `cupy` 做 BGK 空间-时间融合、坡度/平坦度/台阶/不确定性归一化 |
| `src/simulation_env_ros2` | 论文演示用 ROS2 仿真环境，基于 PCD 地形、2D 运动学机器人和模拟雷达 |

当前 FSGP-BGK 包的工程状态：

- `fsgp_bgk` 算法目录不是标准 ROS2 package，没有 `package.xml`、`setup.py` 或 `CMakeLists.txt`。
- `node_ros2.py` 需要从源码目录直接运行，配置路径是相对路径 `../config/params.yaml`。
- 默认输入是 `/simulated_lidar_local` 和 `/odom`，与当前主链路的 `/cloud_registered`、`/Odometry`、`/car_0_terrain_analyzer/fused_map` 不一致。
- 默认输出是 `traversability_pcl` 和 `traversability_pcl_ldd` 两个 `PointCloud2`，不是规划器直接消费的 `SE2Grid`。
- 默认局部地图只有 `x_length=5`、`y_length=5`，更像局部演示，不适合作为当前全链路规划地图。
- GPU 已经被使用，但不是完整 GPU 管线：`torch/cupy` 之外仍有 Open3D、sklearn、NumPy 和 CPU/GPU 数据来回复制。
- 已将上游 ROS1/ROS2 演示仿真包、PCD demo 地图和嵌套 `.git` 元数据移出主 workspace，当前只保留算法源码作为 vendor/reference。

因此它的算法思想可用，但需要工程封装和接口重构。

## 4. 为什么 FSGP-BGK 需要 SEB

用户给出的判断是对的：

```text
原版 FSGP 会为同一个二维位置给出同一个可通行性分数。
SEB 能根据车辆航向把它们区分开。
```

这在越野地形里非常关键。同一个 `(x,y)`，车辆朝向不同，风险可能完全不同：

- 车辆沿着沟壑方向通过，轮子左右落差小；横穿沟壑时容易托底或侧翻。
- 沿坡行驶主要体现俯仰风险；横坡行驶主要体现侧倾风险。
- 通过窄障碍间隙时，车体航向和 footprint 决定是否碰撞。
- 台阶或岩脊对短轴/长轴方向的影响不同，不能用一个二维分数概括。

FSGP-BGK 适合提供平滑高程、局部几何特征和不确定性；SEB 适合在每个 yaw bin 上用车辆几何和姿态约束重新评估风险。两者结合后，输出应是：

```text
risk = risk(x, y, yaw)
```

而不是：

```text
risk = risk(x, y)
```

这也和本工程已有的 `SE2Grid` 数据结构天然一致。

## 5. 推荐集成架构

### 5.1 第一阶段架构：旁路增强节点，不替换主链路

推荐先新增一个 ROS2 package，例如：

```text
src/fsgp_bgk_se2
```

节点名：

```text
fsgp_bgk_se2_node
```

当前分支已经落地第一版 `src/fsgp_bgk_se2`：

- `fsgp_bgk_se2_node`：订阅 `SE2Grid`，发布增强后的 `/car_0_fsgp_bgk/fused_map`。
- GPU 优先：检测到 CuPy/CUDA 时使用 GPU 数组计算。
- CPU 托底：没有 CuPy/CUDA 时自动回退 NumPy。
- `terrain_nav` 已增加 `risk_query_mode=se2_yaw`，可以按 `(x,y,yaw)` 查询增强 `risk`。
- 上游 FSGP-BGK 的演示仿真包和 PCD 资产已从主 workspace 移走，避免污染主 workspace。

这里不是严格的单线结构。更准确的拓扑是：

```text
FAST-LIO2
  |-- /cloud_registered, /Odometry ------------------+
  |                                                   |
  v                                                   v
terrain_analyzer ---------------------------> fsgp_bgk_se2_node
  |                                             ^
  | /car_0_terrain_analyzer/fused_map           |
  +---------------------------------------------+
                                                |
                                                v
                                  /car_0_fsgp_bgk/fused_map
                                                |
                                                v
                                           terrain_nav
```

也就是说，FSGP-BGK/SEB 增强节点可以直接使用 FAST-LIO2 的点云和里程计，但它最好也使用 `terrain_analyzer` 已经整理好的基础地形层。这样它不是被动排在 `terrain_analyzer` 后面，而是作为融合节点站在规划器前面。

输入：

| Topic | 类型 | 说明 |
| --- | --- | --- |
| `/car_0_terrain_analyzer/fused_map` | `se2_grid_msgs/msg/SE2Grid` | 基础高程、方差、有效区域、基础风险 |
| `/cloud_registered` | `sensor_msgs/msg/PointCloud2` | 可选，用于 FSGP 原始点云特征更新 |
| `/Odometry` | `nav_msgs/msg/Odometry` | 用于局部窗口、时间戳和车体姿态上下文 |

输出：

| Topic | 类型 | 说明 |
| --- | --- | --- |
| `/car_0_fsgp_bgk/fused_map` | `se2_grid_msgs/msg/SE2Grid` | 保持规划器兼容的增强版 fused map |
| `/car_0_fsgp_bgk/bgk_traversability_cloud` | `sensor_msgs/msg/PointCloud2` | BGK 二维可通行性调试可视化 |
| `/car_0_fsgp_bgk/seb_risk_cloud` | `sensor_msgs/msg/PointCloud2` | 指定 yaw slice 的 SEB 风险调试可视化 |

增强版 `fused_map` 应保留现有规划器依赖的层：

```text
elevation
var
valid
risk
zbx
zby
```

并新增调试/分析层：

```text
bgk_mean
bgk_var
bgk_traversability
seb_support
seb_pitch_risk
seb_roll_risk
seb_clearance_risk
seb_risk
```

其中 `risk` 作为最终规划风险层，可以由如下方式融合：

```text
risk_final(x,y,yaw) =
  max(
    baseline_risk(x,y),
    seb_risk(x,y,yaw),
    uncertainty_weight * bgk_var(x,y)
  )
```

也可以先用参数控制：

```yaml
risk_fusion_mode: max        # max | weighted
baseline_weight: 0.40
seb_weight: 0.45
uncertainty_weight: 0.15
```

### 5.2 第二阶段架构：规划器读取 yaw 风险

当前 `terrain_nav` 需要做一处关键改造：

1. 保留二维 costmap 用于硬障碍物 clearance 快速计算。
2. 新增 yaw-aware 风险查询函数。
3. A* primitive、direct curve、MPPI rollout、terrain speed scale 都使用 `(x,y,yaw)` 查询。

建议新增参数：

```yaml
risk_query_mode: se2_yaw       # flat_2d | se2_yaw
se2_risk_layer: risk
se2_risk_interpolation: nearest
```

行为：

- `flat_2d`：保持现在逻辑，兼容旧地图。
- `se2_yaw`：如果 `risk` 是 SO2 layer，则按轨迹点 yaw 查询 `risk(x,y,yaw)`；如果不是 SO2 layer，则自动回退到二维风险。

这样可以保证新模块可逐步打开，不会破坏现有 FAST-LIO2 + terrain analyzer 链路。

### 5.3 为什么不一开始放在 terrain_analyzer 前面

把 FSGP-BGK 放在 `terrain_analyzer` 前面也不是不行，但那样它就变成了“地形分析器的一部分”或“地形分析器替代品”，工程改动会明显更大。

当前原版 FSGP-BGK 的 ROS2 节点输入是点云和里程计，输出是 `PointCloud2` 形式的 traversability map。它没有输出本工程规划器需要的 `SE2Grid`，也没有输出 `terrain_nav` 当前依赖的这些层：

```text
elevation
var
valid
risk
zbx
zby
```

而当前 `terrain_analyzer` 除了计算基础地形风险，还做了很多工程化处理：

- 点云到高程图的融合。
- unknown/valid 区域维护。
- 垂直障碍物、树干、墙体等硬障碍检测。
- 车体自身点过滤。
- 障碍物生命周期、膨胀和 clearance 相关风险。
- 发布规划器已经适配好的 `SE2Grid`。

如果把 FSGP-BGK 强行放到它前面，会遇到两个问题：

1. `terrain_analyzer` 当前并不消费 FSGP-BGK 的 traversability 点云，所以还要重写输入接口。
2. 如果直接绕过 `terrain_analyzer`，又会丢掉上述硬障碍和地图维护逻辑，规划器反而更容易误判。

所以推荐顺序是：

- 短期：FSGP-BGK 做 `terrain_analyzer` 的增强层。
- 中期：FSGP-BGK/SEB 输出 yaw-aware `SE2Grid`，让 `terrain_nav` 真正读取 `risk(x,y,yaw)`。
- 长期：如果 FSGP-BGK/SEB 已经稳定覆盖高程、方差、valid、硬障碍和风险层，再考虑把它前移或直接替换 `terrain_analyzer`。

## 6. FSGP-BGK/SEB 算法流程

推荐每帧执行流程如下：

```text
输入 fused_map/cloud/odom
        |
        v
1. 提取局部窗口
        |
        | elevation, var, valid, optional point cloud features
        v
2. FSGP/BGK 回归
        |
        | bgk_mean(x,y), bgk_var(x,y), bgk_traversability(x,y)
        v
3. SEB 航向扫描
        |
        | for every yaw bin:
        |   evaluate vehicle footprint/contact/support
        |   compute pitch/roll/step/clearance risk
        v
4. 风险融合
        |
        | risk(x,y,yaw)
        v
5. 发布增强 SE2Grid 与调试点云
```

SEB 建议评估项：

| 项 | 输入 | 输出 |
| --- | --- | --- |
| 纵向坡度风险 | 车体 x 轴方向高程差/法向量 | `seb_pitch_risk` |
| 横向坡度风险 | 车体 y 轴方向高程差/法向量 | `seb_roll_risk` |
| 车体支撑风险 | 轮胎/footprint 采样点有效比例 | `seb_support` |
| 台阶/托底风险 | footprint 内最大局部高差 | `seb_step_risk` |
| 障碍 clearance 风险 | baseline hard obstacle/vertical seed | `seb_clearance_risk` |

本工程已有的老 CUDA 文件 `src/uneven_mapping/terrain_analyzer/src/TerrainBank.cu` 中，`compute_map_se2()` 已经有类似思路：它对每个 `(cell,yaw)` 使用车体椭圆 footprint 采样地形，估计局部法向量、曲率和纵横向坡角 `phx/phy`，再写入 yaw-dependent `risk`。这部分可以作为 SEB 实现参考。

## 7. GPU 落地方案

### 7.1 当前 GPU 现状

工程里目前有两条 GPU 线索：

1. `FSGP-BGK-ROS2` 使用 `torch/gpytorch/cupy`，但仍有 CPU 下采样、sklearn KNN、NumPy 拼接和 ROS message 拷贝。
2. `terrain_analyzer` 目录里已有 `terrain_analyzer_gpu_node.cpp` 与 `TerrainBank.cu`，但当前 `CMakeLists.txt` 只编译 CPU 版 `terrain_analyzer`。

所以“地形分析与高斯回归都跑 GPU”是可行的，但建议分两步做：

### 7.2 近期方案：Python GPU 原型

用于快速验证算法效果：

- 把 `fsgp_bgk` 封装成标准 `ament_python` ROS2 package。
- 将配置路径改为 ROS2 parameter，不再依赖相对目录。
- 输入适配当前 topic。
- 输出 `SE2Grid`，而不是只发布 `PointCloud2`。
- 保持 `torch/gpytorch/cupy` GPU 推理。

需要优化的点：

- 用 `torch` 或 `cupy` 替代 Open3D/sklearn 的关键循环。
- 避免每帧重新完整训练 GP；可用滑动窗口增量样本、低频训练、高频预测。
- 减少 CPU/GPU 之间的往返复制，只在 ROS 发布前 copy 最终 layer。

### 7.3 中期方案：C++/CUDA 工程化

用于稳定部署和实时性：

- 启用 `terrain_analyzer_gpu_node.cpp + TerrainBank.cu`，先恢复已有 yaw-dependent `risk` 能力。
- 将 FSGP/BGK 的核心矩阵运算迁移为 CUDA/C++ 或 LibTorch CUDA。
- 将 SEB yaw sweep 写成 CUDA kernel，对 `(cell,yaw)` 并行计算。

推荐 GPU kernel 拆分：

| Kernel | 功能 |
| --- | --- |
| `point_cloud_to_grid_kernel` | 点云栅格化，更新高程/方差/计数 |
| `normal_curvature_kernel` | 计算法向量、坡度、曲率、台阶 |
| `bgk_predict_kernel` | 对局部网格做 BGK/SGP 预测与方差估计 |
| `seb_yaw_sweep_kernel` | 对每个 `(x,y,yaw)` 评估 footprint 风险 |
| `risk_fusion_kernel` | 融合 baseline/BGK/SEB 风险 |

ROS2 消息仍然在 CPU 侧发布，GPU 侧只负责计算。

## 8. 建议实施步骤

### Phase 0：源码接入卫生

- 决定 `src/FSGP-BGK-ROS2` 是 submodule 还是 vendor 源码。
- 如果作为 vendor，移除嵌套 `.git`。
- 保持不用的 ROS1/ROS2 演示仿真包不进入主 workspace，避免混编干扰。
- 明确 Python 依赖版本，尤其是 CUDA 对应的 `torch` 与 `cupy`。

### Phase 1：只做离线对齐验证

- 已完成第一版主链路 adapter：输入 `/car_0_terrain_analyzer/fused_map`，输出 `/car_0_fsgp_bgk/fused_map`。
- 当前先使用 BGK 风格局部 Gaussian 平滑，不直接每帧训练原版 GP，保证实时性和可回退性。
- 后续再把 `/cloud_registered`、`/Odometry` 接入原版 FSGP 训练/预测路径。

### Phase 2：输出 SE2Grid

- 已完成新节点输出 `/car_0_fsgp_bgk/fused_map`。
- 已加入 SEB yaw sweep，输出 `risk`、`seb_risk`、`seb_pitch_risk`、`seb_roll_risk`、`seb_step_risk`、`seb_support` 等 SO2 layers。
- 需要在 RViz 中继续对不同 `yaw_index` 的 risk slice 做可视化对比。

### Phase 3：规划器使用 SE2 风险

- 已完成 `terrain_nav` 参数 `risk_query_mode=se2_yaw`。
- footprint cost、terrain speed scale、direct corridor sampling 已改为在有 SO2 risk layer 时按 yaw 查询。
- 二维 costmap 仍用于硬障碍和 clearance，避免某个航向的软风险把整个 `(x,y)` 永久打成硬障碍。

### Phase 4：GPU 性能优化

- 原型阶段目标：`fsgp_bgk_se2_node` 5-10 Hz。
- 规划器目标：20-25 Hz 控制周期不被地图更新阻塞。
- 如果 Python 原型 CPU/GPU 往返开销过大，再迁移核心计算到 C++/CUDA。

### Phase 5：仿真验证

验证场景：

- 平地直行，风险应低且无航向差异。
- 横坡/纵坡，同一位置不同 yaw 的 `seb_roll_risk`/`seb_pitch_risk` 应明显不同。
- 沟壑/岩脊，沿向和横向通过风险应不同。
- 树干/墙体密集障碍，clearance 与 hard obstacle 仍应可靠。
- 下坡转弯，检查 FAST-LIO 漂移是否来自 SLAM/IMU/点云，而不是风险层。

## 9. 参数建议

初始建议配置：

```yaml
fsgp_bgk_se2_node:
  ros__parameters:
    use_sim_time: true
    input_fused_map_topic: /car_0_terrain_analyzer/fused_map
    input_cloud_topic: /cloud_registered
    odom_topic: /Odometry
    output_fused_map_topic: /car_0_fsgp_bgk/fused_map
    frame_id: map

    map_resolution: 0.20
    local_length_x: 18.0
    local_length_y: 18.0
    yaw_bins: 72
    publish_rate_hz: 5.0

    use_gpu: true
    gpu_backend: torch_cupy
    inducing_points: 500
    lengthscale: 0.10
    alpha: 10.0

    risk_fusion_mode: max
    baseline_weight: 0.40
    seb_weight: 0.45
    uncertainty_weight: 0.15

    vehicle:
      footprint_front: 0.42
      footprint_back: 0.42
      footprint_half_width: 0.313
      footprint_margin: 0.10
      max_pitch_rad: 0.55
      max_roll_rad: 0.45
      max_step_height_m: 0.16
      min_support_ratio: 0.70
```

规划器侧：

```yaml
terrain_nav_node:
  ros__parameters:
    fused_map_topic: /car_0_fsgp_bgk/fused_map
    risk_query_mode: se2_yaw
    se2_risk_layer: risk
    hard_risk_layer: baseline_risk
```

## 10. 风险与注意事项

1. FSGP-BGK 不能直接解决 FAST-LIO 漂移。漂移要继续检查雷达点云数量、IMU 频率、时间戳、外参、车体动力学和仿真物理。
2. 原版 FSGP-BGK 的 ROS2 节点是演示性质，不是当前工程可直接消费的接口。
3. 每帧训练 GP 成本高，必须控制局部窗口、诱导点数量和更新频率。
4. Python GPU 原型容易被 CPU/GPU 数据拷贝拖慢，部署前要做 profiling。
5. 若 `terrain_nav` 不改 yaw-aware 查询，即使 FSGP-BGK 输出了 SE2 风险，也会在规划器里被压回二维，核心价值会丢掉。

## 11. 推荐最终形态

推荐最终链路如下：

```text
FAST-LIO2
  |-- /cloud_registered, /Odometry -> terrain_analyzer -> /car_0_terrain_analyzer/fused_map --+
  |                                                                                           |
  +-- /cloud_registered, /Odometry -----------------------------------------------------------+
                                                                                              v
                                                                                       fsgp_bgk_se2_node
                                                                                              |
                                                                                              v
                                                                                  /car_0_fsgp_bgk/fused_map
                                                                                              |
                                                                                              v
                                                                               terrain_nav(risk_query_mode=se2_yaw)
                                                                                              |
                                                                                              v
                                                                                       /hunter_se/cmd_vel
```

这样保留了当前已经调通的建图/仿真基础，也能把 FSGP-BGK 的概率地形估计和 SEB 的航向相关可通行性接入到真正的规划控制闭环里。
