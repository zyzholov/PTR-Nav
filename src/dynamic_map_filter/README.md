# 动态障碍物建图辅助模块

本包封装官方 DUFOMap/UFOMap 和 FreeDOM 的原生 C++ 核心，用于维护 FAST-LIO 的**静态累计地图和注册树**。地形分析、实时碰撞检查继续接收原始当前扫描；辅助模块不能把未知空间声明为可通行地面。

## 源码及修改

| 后端 | 官方源码 | 固定版本 | 本工程处理 |
|---|---|---|---|
| `dufomap` | https://github.com/KTH-RPL/dufomap | `9e239ddd5995136e14f5212f33382a6ebc59e518`；UFOMap `fcb20f445d257280c039c07ab756c9d5b37ebd78` | 官方头文件未修改；通过 C++17 PIMPL 接入私有 C++20 核心 |
| `freedom` | https://github.com/LC-Robotics/FreeDOM | `dcfd5690cafddf121a80bd8ea477807d9656748e` | 去除 ROS1 包装依赖；修正历史地图第二层关联的反向条件。所有修改及许可证均保存在 `vendor/freedom/` |

FreeDOM 的实质修正见 `HISTORY_RING_FIX.patch`：`MRMap::map_removal` 中，`if (!cluster_aggressive_twice)` 与配置含义及 `scan_removal` 的执行条件相反，导致 124 邻域配置缺失最后一层关联。回归用例在修改前失败、修改后通过，并检查断开的点不被关联。这里使用的是**本地修正版**，不能作为原论文未经修改的对比结果。

后续射线修正见 `MEASURED_RAYS_AND_BOUNDARIES.patch`：采用同一帧实际去畸变回波而非体素质心，修正起始格面与同时穿越多格面的边界处理。当前占据仍截断射线，同帧多射线不增加时间票数；独立的线段/体素相交检查已通过。FAST-LIO 接口还按既有 Float32 分类标签的实际取值区间定位历史点，避免用“整数标签×分辨率”反算造成边缘漏删。分类、地面保护、连续证据门槛与原始导航输入均保留。原生正确性不等于整链路残影清理时限已通过，物理资格须单独记录。

## 数据边界

```mermaid
flowchart LR
  S[倾斜25° MID360 + 外置IMU] --> L[FAST-LIO 位姿与当前扫描]
  L --> C[cloud_registered 原始当前扫描]
  C --> T[GPU地形分析 → GP补全/共享风险 → SE2规划/跟踪]
  L --> H[DUFOMap 或 FreeDOM 历史动态点分类]
  H --> M[静态累计 Laser_map 与注册IKD树]
  M -. 仿真资格配置中仅作 advisory 记忆 .-> T
```

两个辅助后端互斥。启动参数是 `mapping.dynamic_filter_backend`，取 `none`、`dufomap`、`freedom`。实时 `/cloud_registered`、`/cloud_registered_visual` 均在历史分类之前发布，XYZ、点数、时间戳不因辅助分类而改变。分类只决定历史点是否进入/留在累计地图和 IKD 树。

* DUFOMap 依靠真实回波射线确认自由体素。没有背景回波、被遮挡、空帧均不能单靠时间变成自由空间。
* FreeDOM 使用带同一扫描关联的空间动态标签。新标签还必须通过本工程的实测表面参考保护条件；该参考保留最多 3 秒，以免车身遮挡立刻丢失保护地面所需的测量。老参考每秒上调保护高度 0.01 m，过期、时钟回退或超过 5 万格预算则丢弃。地面以下或无参考的历史点不因分类直接删除。记住地面本身不产生自由空间；已接受标签在上游撤销分类后撤销。
* FreeDOM 深度补全版本仅作独立回放诊断，生产 `freedom` 后端关闭该功能；没有将推测的“天空自由射线”接入导航。
* 实时碰撞层仍保留当前运动物体。不能用清理后的静态地图替代当前碰撞输入。

## 参数与资源

DUFOMap：0.10 m 体素、0.20 m 命中保护、2 格未知保护、最大 50 m、TBB 2 线程、400 万节点预算。FreeDOM：0.10 m 子体素、官方 2/5 层级（0.40 m 分类体素、3.20 m 块）、26/124 关联、连续 3 次自由证据、20 次占据撤销自由、40 m 局部范围、30 m 射线、2 线程，静态体素 30 万预算。预算重置不声明自由；当前原始数据始终继续发布。FreeDOM 内部静态结构使用全局层级参数，不应在一个进程内同时实例化不同层级。

两个原生辅助后端运行于 CPU。地形特征和共享 GP 的 CUDA/CuPy 路径独立保留；没有把这些毫秒级辅助核宣称为 GPU 实现。

## 启动与检查

```bash
scripts/sim/start_mapping.sh rviz:=false dynamic_filter_backend:=freedom
```

其余三个模块分别按 `docs/MID360_TILT25_QUALIFICATION_20261027.md` 启动；仿真中已验证的地形链路使用 `obstacle_memory_role:=advisory`，不能将旧累计障碍点无条件重新写成硬碰撞。

```bash
build/dynamic_map_filter/test_dufomap_filter --gtest_color=no
build/dynamic_map_filter/test_freedom_filter --gtest_color=no
# 导出真实记录；必须逐帧同时间戳匹配位姿，不插值未来位姿，不输入仿真真值
python3 scripts/sim/checks/export_mapping_replay.py RUN/mapping_inputs OUTPUT.bin
# ROI/floor/vacate 参数只用于输出统计，不传给算法
build/dynamic_map_filter/replay_filter OUTPUT.bin freedom stats.csv 5 0 -.47533 57.78
```

离线回放检查分类效果，不能替代包含车辆物理、定位漂移、通信延迟和跟踪响应的闭环资格测试。小规模资格记录见 `docs/DYNAMIC_MAPPING_NAV_QUALIFICATION_20261027.md`；大批量对比实验尚未启动。
