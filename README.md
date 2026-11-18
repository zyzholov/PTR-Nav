# PTR-Nav

PTR-Nav 是一个面向复杂非结构化地形的 ROS 2 UGV 导航、建图、仿真与规划控制工程。当前分支记录 Isaac Sim、FAST-LIO2、GPU 地形分析、FSGP-BGK/SEB 增强与规划控制的集成开发状态，保留源码、配置、模型和脚本，不提交 `build/`、`install/`、`log/` 等本地编译/运行产物。

## 当前分支状态

动态避障当前先限定到**单个可见移动物体**。横穿/迎面已有到达记录，但重复资格仍暴露停车后受撞、动作中断及交付问题，不能声称稳定可靠。未通过的净距偏好和保守调度试验已撤回；测试现校验同域独占，并在首个失败后停止。最新逐版本失败、撤回和资格目录见 [动态避障修正记录](docs/DYNAMIC_INTERACTION_REPAIR_20261109.md)。多行人及复杂遮挡封门暂不作为已完成能力。

2026-11-09（分版本资格，不表示当前全部通过）：已修正本机 MID360 渲染回波的采集时间关联、外置 IMU 偏心加速度及观察器的源时间配对，并加入提前丢弃预计来不及交付的旧地图调度。当前采集时钟下，密集柱阵 39.54 秒到达停车、第一排 19.26 秒通过；9.58°上/下坡分别 18.08/17.30 秒通过；实际横穿与三动态组合分别 32.40/40.91 秒通过，累计旧障碍点 0.24 秒后持续清理达标。运动风险图 5.58～6.23 Hz，最大交付源龄≤0.30秒。6厘米台沿 50.125 秒到达且时效合格，但原整轮终态截图失败，保留失败记录。预设输入源龄窗口为0.16秒，输出期限仍0.30秒；未改变尺寸、风险预算或碰撞余量。**未完成所有能力资格，有限定向通过不是通用成功率或实车概率安全保证；修正前没有计入一帧真实采集时延的历史结果不与本版混合。**后续 R3 形状配准速度修正使柱阵 35.34 秒到达，横穿 24.75 秒到达，但封门回退仍超时；R4 恢复流程物理封门仍失败，不能继承 R1/R3 资格。扩展场景及稳定性仍在复核。四模块命令见 [NEOS 启动说明](docs/NEOS_SIM_STARTUP.md)，本轮原理、证据及失败记录见 [采集时钟与导航资格修正](docs/NEOS_SOURCE_CLOCK_QUALIFICATION_20261109.md)，历史记录见 [NEOS 资格记录](docs/NEOS_MODEL_QUALIFICATION_20261105.md)。

2026-10-27：默认扫描为 **MID360 倾斜 25°、40 阶段 CSV 方向表、真实 RTX 回波**，已修正内部坐标重复转换导致的畸变。本轮接入本地修正的 FreeDOM 建图辅助核、短时实测参考保护，并约束共享 GP 的二次曲率外推。清理复测分别在 0.95/0.70 秒达到旧障碍点减少至少 80% 的门槛；持续占道绕行及 0.45 米抬升坡道横穿取得闭环到达记录。仍有少量残影，坡道跟踪误差较大，不声明通用成功率；RoboSense 32 线尚未接入。见 [动态建图与导航联调记录](docs/DYNAMIC_MAPPING_NAV_QUALIFICATION_20261027.md)、[MID360 几何与启动记录](docs/MID360_TILT25_QUALIFICATION_20261027.md)。

此前水平 Ouster 替代配置的七个定向用例通过了**位置目标到达/停车或预期拒行**检查，包括 Mars 往返、连续绕障、上下坡动态横穿和约 18.7° 下坡；这些记录不自动适用于新 MID360 配置。终点航向尚未完整接通验收，不能当作 SE(2) 终点姿态通过。[历史逐次数据与图表](results/navigation_completion_20261027/README.md)保留先前失败，不以有限用例声明通用成功率。

<details>
<summary>历史集成与实验记录（不能代替当前版本资格）</summary>

2026-10-26 后续修订：解耦控制地图更新与后台搜索锁；新增独立全分辨率点云显示，保持导航降采样不变；风险曲面贴合实测/合格补全高程，加入原始占据轮廓与距离着色回波。历史障碍只作为软选路偏好时，三柱绕障两次到达；下坡动态冲突也到达，但最小净距仅 0.110 米，不能称为余量充分。保留未开启软选路的停车失败及全部时效数据，见[显示与导航修订记录](docs/NAVIGATION_DISPLAY_QUALIFICATION_20261026.md)。规划长尾和动态余量仍需优化，不宣称通用导航已完成。

2026-10-26：碰撞几何修正后已执行整链物理资格，并据失败修正当前扫描发布顺序和动态提前制动。最终同版 8 次定向检查中 7 次完整通过（包含台阶拒行负例）；另一次虽到达，但风险图源年龄超限，仍保留为失败。平路及上下坡横穿、连续绕障、默认 Mars GUI 短程往返均有实际记录；约 240 毫秒规划长尾及偶发时效抖动尚未解决，不能宣称通用高成功率。见[实时性与动态避障实测](docs/NAVIGATION_RUNTIME_QUALIFICATION_20261026.md)、[碰撞几何与投影契约](docs/COLLISION_GEOMETRY_CONTRACT_20261026.md)。

2026-10-24：继续修正共享地形概率界、实测地面/柱顶角色、细障碍保持、有界短退让与倒车恢复衔接，并加入 GPU 仿真地图时效保护。最终同一运行版本通过退让后恢复、上下坡动态横穿、较陡侧向目标下坡的定向检查；三柱绕行仍未到达，较陡下坡仍有更新长尾，不能宣称通用导航已全部达标。444 项 Python 与 100 项原生导航测试通过。当前原理、完整失败审计、图表和复现命令见[地形与非平坦导航修正记录](docs/TERRAIN_NAV_REFINEMENT_20261024.md)。未修改本轮 FAST-LIO2 配置或降采样，未推送 GitHub。

2026-10-23 非平坦专项继续修正了曲线车身朝向、短观测路径优先级、动态横穿谨慎推进及换向制动包络。当前同一二进制完成连续绕障、平路横穿、坡面上/下行横穿、18.65° 轴向及侧向目标下坡、断流停车恢复的定向资格；较陡下坡仍需约 43..45 秒，动态最小净距余量也偏小，不能声称通用非平坦导航已达标。最新修订和逐次失败见[非平坦导航资格记录](docs/NONFLAT_NAVIGATION_20261023.md)，不要将不同修订混算成功率。

2026-10-22：共享地形后验分支已取得静态绕柱、动态横穿后恢复行驶、上坡及默认Mars短程前进/倒车返回的整链资格记录。修复了障碍物污染地面、体素过滤后旧否决无法清除，以及引导目标搜索阻塞控制的问题。启动时须显式选择下文的 `shared_gp` 和固定仿真起始假设；旧边际配置不会自动切换。**这些是限定场景检查，尚不足以声称全场景安全或高成功率**。当前仅获准小规模资格测试，批量实验仍待审阅。见[接口与公式](docs/PROBABILISTIC_NAVIGATION.md)、[逐次资格记录](docs/QUALIFICATION_20261020.md)和[研究问题/外部对比/消融审查稿](docs/RESEARCH_AND_VALIDATION_PLAN.md)。

最新修订统一了地形与规划的实际足迹支撑集合，使用共享高程误差事件预算。后台GP候选仅供显示；仿真默认启用的导航补全是另外经过当前扫描、距离、包围性和模型方差门控的条件GP，不能把全部背景预测都当作导航支撑。RViz可切换风险曲面、独立实测/预测高程、累计点云和来源。启动区域默认未知；固定初始支撑假设仅获准在仿真中显式验证，不写成实测、不随车扩张，实车禁用。配置与模型边界见[初始支撑契约](docs/PROBABILISTIC_NAVIGATION.md#51-显式仿真初始支撑假设)。后方未观测足迹仍会阻止直接倒车，不用安全拒绝冒充目标到达。

本轮场景扩展前的基线完成以下七类定向资格，51个运行源码、配置和二进制哈希一致。
每类是一次限定检查，不是批量成功率统计；均使用显式共享模型和固定仿真初始假设。

| 检查 | 结果 | 关键观测 |
|---|---|---|
| 7米静态绕柱 | 通过 | 位置误差0.227米，最小采样车身净距0.506米 |
| 9米近距离动态横穿 | 通过 | 误差0.203米，最小采样净距0.237米，停车响应0.725秒 |
| 障碍物离开后的清除与通行 | 通过 | 固定道路姿态概率支撑0.46秒恢复，累计图残留0.86秒达到清除阈值 |
| 0.5米高台阶 | 安全拒行 | 前进约1.99米后停止，距前壁采样净距1.227米 |
| 默认Mars前进/倒车返回 | 通过 | 位置误差0.246/0.235米 |
| 7米上坡 | 通过 | 爬升0.452米，位置误差0.181米 |
| 7米下坡 | 通过 | 下降0.449米，位置误差0.207米 |

运动窗口风险图约7.8至8.0Hz，源年龄P95为0.20至0.285秒、最大约0.305秒。
252项Python、45项原生导航回归通过，另有5组真实CUDA前端检查和5组补全到
原生规划器检查。净距为采样真值几何检查；累计点云清除阈值不是“全部点清零”。
完整定义、失败记录和各次目录见[资格记录第11.37节](docs/QUALIFICATION_20261020.md)。
本地机器可读清单为 `log/navigation_final_qualification_20261022.json`。

新增连续错列障碍、双墙通道、坡面绕障的受控物理场景及逐障碍车身净距验收。
扩展中发现并修正实时障碍重复边界导致的锁死，以及估计坐标系终点门限过松的问题。
新版本不能直接继承上表旧版本资格；当前修正、重复测试、失败记录与待覆盖范围见
[场景扩展记录](docs/NAVIGATION_SCENARIO_EXPANSION.md)。
扩展中已取得三类组合场景的到达/停车记录，以及动态横穿、台阶拒行、
Mars双侧短程目标和倒车返回记录。仿真发布现默认使用异步传输；
280项Python、47项原生回归通过。坡面近障效率、GUI自动截图和更多场景覆盖
仍待改善，不能由有限资格记录宣称通用安全成功率。

本次版本分支为 `isaac-gpu-nav-baseline-20261010`，基于 `fsgp-bgk-se2-integration` 保存当前集成与实验基线，不修改仓库默认分支，也不代表已完成全场景验证。

实验只记录本工程方法，尚未运行外部对比，也未完成最新版本全场景物理资格验证。主阅读入口为[地形预测、航向风险与导航综合实验报告](docs/reports/comprehensive_study/PTR_Nav_Comprehensive_Study.pdf)，模块启动见[脚本目录说明](scripts/README.md)。

当前重点链路：

- ROS 2 Humble 工作空间，使用 `colcon` 构建。
- FAST-LIO / FAST-LIVO2 相关源码和 Hunter/MID360/D455 仿真配置。
- Gazebo Hunter SE + Livox MID360 仿真链路。
- Isaac Sim Hunter SE + RTX LiDAR + 外置 IMU + RealSense D455 链路；默认 MID360 25° 使用有限循环 CSV 扫描表，不声称真实光学机制或逐点运动时序等价。
- 地形分析、SE2 栅格、车式格点规划、默认混合跟踪和可选随机采样 MPC。当前跟踪不是标准 MPPI。

### 系统性实验

当前报告按缺测预测、噪声、离散化、GP风险接入、计算性能及规划/模型闭环六个问题组织，明确区分人工隐藏、自然缺测、无参考与候选拒绝。模型闭环和Isaac接触动力学不是一项测试，不与组件成绩混算。

历史记录含 20,000 个地形条件、1,400 个模型任务和小批次 Isaac 物理记录，不属于当前修正版的验证量。当时物理绕障、增强风险及累计地图清除仍有问题，补全默认关闭；当前仿真条件分支的有限资格结果另见上述最新记录。重绘图表不是重新跑实验，也不声明方法优于外部算法。

当前 PTR-COMPREHENSIVE-04R1 已执行500输入的4000次缺测尝试（3995完成、5输入不足）、3600噪声条件、720离散配置、540次CPU/GPU正式计时、1000查询的3000次原生规划调用、420模型闭环任务。另以300共享拟合计算900次风险接入，全部保留原始数组/路径与失败。规划主批次使用实测高程，补全关闭。

模型闭环275安全到达、120安全拒绝、25未完成（24突现障碍、1绕障），未记录碰撞/地形越界。风险接入中实测修正、有界补全仍新增危险漏判，不能据误拒改善开启部署；窄脊与细柱完整观测仍有离散漏判。205项回归通过不等于全场景验收。

此前定向排查确认旧Swiss参数配置及KITTI第08序列部分子图注册存在问题；GP支撑判断已加入凸包与角度包围约束。详见[修正附件](docs/reports/systematic_study/PTR_Nav_Terrain_Correction.pdf)。旧6000次及旧1000查询地图保留，旧3000次调用计划未执行。当前批次仍暴露窄脊、台阶及细柱漏判，不能声称全工程达标。

[主报告与实验协议](docs/reports/comprehensive_study/README.md) | [历史附件索引](docs/reports/README.md)

```bash
bash scripts/sim/experiments/build_systematic_report.sh --comprehensive
```

</details>

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

修改地形Python源码后，也需要更新ROS实际加载的安装目录：

```bash
colcon build --packages-select fsgp_bgk_se2
python3 scripts/tools/check_runtime_installation.py
```

启动地形模块时会核对Python模块及launch/config/RViz文件内容；旧安装副本与源码
不一致时会提示重建。仅源码单测通过不能证明ROS已运行新代码。

如果只调 FAST-LIO：

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select fast_lio --symlink-install
source install/setup.bash
```

## Isaac Sim 模块化链路

当前仿真链路不再使用“大一键脚本”把所有节点绑在一起。按下面的顺序分别启动，每个终端只负责一个模块：

四个脚本统一使用 Fast DDS 异步发布和项目的大消息共享内存配置；已有的
`FASTRTPS_DEFAULT_PROFILES_FILE` 会优先保留。`HUNTER_ISAAC_DDS_PROFILE=builtin`
可关闭项目的自动配置选择，供传输排查使用。手动执行 ROS CLI 时，每个终端先运行：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="$PWD/scripts/sim/config/fastdds_large_maps.xml"
```

各模块和检查必须使用相同 `ROS_DOMAIN_ID`；不要沿用其他工程的 Cyclone DDS 环境观察这条链路。

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
scripts/sim/start_mapping.sh rviz:=false dynamic_filter_backend:=freedom
```

Terminal 3，GPU 地形分析前端 + FSGP-BGK/SEB 增强：

```bash
scripts/sim/start_terrain_analysis.sh terrain_backend:=gpu fsgp_gpu_backend:=cupy \
  prediction_backend:=feature_sgp gp_device:=cuda \
  probability_model:=shared_gp gp_complete_supported:=true \
  probability_gp_max_distance_m:=1.25 probability_gp_max_extrapolation_m:=0.75 \
  startup_support_mode:=stable_stance startup_support_directional:=true \
  obstacle_memory_topic:=/Laser_map obstacle_memory_role:=advisory
```

Terminal 4，规划与底盘控制：

```bash
scripts/sim/start_planning_control.sh chance_allow_gp_completion:=true \
  chance_allow_joint_model:=true chance_allow_startup_prior:=true reverse_recovery_enable:=true
```

上面是当前仿真资格验证用法：`stable_stance` 只在通过稳定性与周边地面一致性检查后，
对固定起始足迹使用显式支撑假设；该范围不随车移动，也不是实测地面。
共享RViz由Terminal 3启动，规划启动后仍使用同一窗口；右侧模式可切换
概率风险、实测/预测高程、累计建图点云和观测来源。
地形脚本默认仍为 `startup_support_mode:=unknown`，规划默认不授权起始假设；
选择完全未知模式时，车身下方未获得合格支撑可能导致等待而无法起步。
实车禁止该仿真假设。当前有限资格结果不能代替全场景导航验收。
1.25 米是实测支撑距离上限；额外显式启用的 0.75 米外推仍保留实际后验协方差、原始矛盾证据与整车概率检查。`startup_support_directional` 仅为倾斜雷达的单侧观测增加稳定姿态/地面一致性检查，不会将前方未知直接放行。需要历史软选路时，给 Terminal 3 加 `obstacle_memory_topic:=/Laser_map obstacle_memory_role:=advisory`；它不提供地面支撑，也不成为硬障碍。

手动控制时不要启动 Terminal 4，改为：

```bash
scripts/sim/start_keyboard_control.sh
```

仅验证传感器和建图时，可以只启动前两个模块。无 GPU 时将 Terminal 3 改为：

```bash
scripts/sim/start_terrain_analysis.sh terrain_backend:=cpu fsgp_gpu_backend:=cpu \
  prediction_backend:=feature_sgp gp_device:=cpu \
  probability_model:=shared_gp gp_complete_supported:=true \
  probability_gp_max_distance_m:=1.25 probability_gp_max_extrapolation_m:=0.75 \
  startup_support_mode:=stable_stance startup_support_directional:=true
```

CPU入口用于兼容和诊断，尚未获得与GPU相同的整运动时效资格。

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
- `/cloud_registered_visual`：同一校正位姿的完整当前扫描，供真实回波显示、Isaac 实时碰撞/运动检测与净空地面参考；现在与注册扫描一起提前发布，不等待累计地图维护，也不把它变成额外 GP 支撑。其他建图器没有此输出时，给规划脚本显式传入 `live_cloud_topic:=/cloud_registered`。
- `/Laser_map`：经过体素去重的累计增量地图，当前 Isaac 配置以 1 Hz 发布，RViz 默认显示此话题；首次初始化后需要等到下一帧完成匹配。
- `/livox/lidar`：仿真雷达原始实时点云，RViz 默认关闭，排查传感器时可手动打开。

## FSGP-BGK / SEB 地形增强链路

`fsgp_bgk_se2` 订阅前端的 `SE2Grid` 和当前注册点云，发布启发式地形层及独立的整车概率层。前述共享分支用同一份局部高程后验计算航向相关几何事件，再供搜索、重规划和制动预算使用。以下省略参数的示例用于单模块观察；自主导航使用上文Terminal 3/4的完整配套配置。

单独观察地形模块：

```bash
scripts/sim/start_terrain_analysis.sh
```

默认 `fsgp_gpu_backend:=auto` 会按计算规模选择后端：当前 `104x104x36` 地图会走 CuPy，仅很小的地图保留 CPU 以避开 kernel 启动开销。强制 GPU 写法是：

```bash
scripts/sim/start_terrain_analysis.sh terrain_backend:=gpu fsgp_gpu_backend:=cupy
```

只需基础地形图时设 `enhance:=false`。若确实需要在两个终端分别调试前端与增强节点，再运行 `scripts/sim/optional/start_fsgp_bgk.sh gpu_backend:=cupy`。

自主导航由上文Terminal 4消费增强地图并独占 `/cmd_vel`。共享地图需要显式的
`chance_allow_joint_model:=true`；省略配套授权会拒绝该地图，不能与普通边际配置混用。

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
抢占话题，不是重复计算同一件事。默认后端保留 BGK 风格平滑；实验性
`prediction_backend:=feature_sgp` 现已使用 GPyTorch/CUDA 特征稀疏 GP 和观测约束局部条件分支，
但不等同于原论文的完整算法复现。仿真脚本现默认启用当前帧条件GP窄缺测补全，
不覆盖实测；实车及通用节点默认仍关闭。已通过窄缺测带的真实规划器资格检查和
短程Isaac前进/倒车停车检查，不代表全场景安全认证，详见[资格记录](docs/QUALIFICATION_20261020.md)。
前端 `risk` 是车辆姿态风险，增强时不再重复足迹膨胀；
规划二维硬障碍使用独立的 `collision_risk`，姿态风险仍按航向查询。

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
