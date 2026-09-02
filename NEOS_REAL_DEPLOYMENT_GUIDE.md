# NEOS 实物部署与规划控制接入操作文档

本文面向当前 `seb_clean` 工程从 Gazebo Hunter 仿真迁移到真实 NEOS 底盘。结论基于当前工程代码、`GAZEBO_FASTLIO_TERRAIN_PLANNING_DEBUG_GUIDE.md`、`GAZEBO_FASTLIO_TERRAIN_PLANNING_NEXT_CONTEXT.md`，以及真实底盘驱动工作空间：

```bash
/home/z/workspace_zyz/driver/hexman/sdk_neos_ws
```

当前仿真链路是：

```text
Livox MID360 + IMU -> FAST_LIO2
  /Odometry, /cloud_registered, /cloud_local_map
-> terrain_analyzer
-> terrain_nav
-> cmd_vel
```

实车迁移不是只改一个 `cmd_topic`。必须同时处理底盘控制语义、车辆几何、坐标系、FAST_LIO 外参、地形阈值和安全限速。

## 1. 当前 NEOS 驱动事实

真实底盘驱动不是 `tank_sdk`，而是 `sdk_neos_ws` 中的 `xpkg_vehicle`。

关键文件：

```bash
/home/z/workspace_zyz/driver/hexman/sdk_neos_ws/src/drivers/xpkg_vehicle/launch/ROS2/xnode_vehicle_test.launch.py
/home/z/workspace_zyz/driver/hexman/sdk_neos_ws/src/drivers/xpkg_vehicle/src/ROS_INF/ros2_interface.cpp
/home/z/workspace_zyz/driver/hexman/sdk_neos_ws/src/drivers/xpkg_vehicle/src/FUNC/vehicle_func.cpp
/home/z/workspace_zyz/driver/hexman/sdk_neos_ws/src/drivers/xpkg_vehicle/src/FUNC/vehicle_func_odom.cpp
/home/z/workspace_zyz/driver/hexman/sdk_neos_ws/src/drivers/xpkg_vehicle/ini/device_id_list.ini
```

驱动接口：

- `xnode_vehicle` 固定订阅 `/cmd_vel` 和 `/cmd_vel_stamp`。
- `/cmd_vel.linear.x` 被作为纵向速度发送给底盘。
- `/cmd_vel.angular.z` 被作为 `rotate` 字段发送给底盘。
- `xnode_vehicle` 发布 `/odom`，frame 是 `odom -> base_link`。
- launch 中已经设置 `pub_tf: False`，这对和 FAST_LIO 共存是正确的。
- `xpkg_vehicle` 必须配合 `xpkg_comm` 运行。

驱动 ini 中 NEOS 配置为：

```ini
[0x0B]
type=NEOS
model=Ackermann
width=0.547
long=0.535
```

注意：你提供的图纸中侧视轴距标注为 `430 mm`，而 SDK ini 的 `long=0.535` 在代码里被当作 Ackermann `wheel_base` 使用。这是必须实测核对的高优先级问题。规划和 MPC 应使用真实前后轮轴中心距；如果实际轴距是 `0.430 m`，不要沿用仿真 Hunter 的 `0.55 m`。

## 2. 底盘命令语义必须实测

当前 MPC 内部计算的是 Ackermann 前轮转角 `steer`。发布 `/cmd_vel` 时由参数决定：

```yaml
mpc/cmd_is_yaw_rate: false  # angular.z = 前轮转角
mpc/cmd_is_yaw_rate: true   # angular.z = 车体 yaw rate
```

NEOS 在 SDK 中是 `Ackermann`，并且 odom 代码在 Ackermann 分支中把 `speed_r` 当作 `steering` 使用。因此初步判断：真实 NEOS 驱动更可能期望 `/cmd_vel.angular.z` 是转角量，而不是 yaw rate。也就是实车初始配置建议保持：

```yaml
mpc/cmd_is_yaw_rate: false
```

但这个判断不能代替实测。第一次上车前必须低速验证：

```bash
# 先让车轮离地，或确保车前方有足够空旷区域。
ros2 topic pub -r 5 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.10, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"

ros2 topic pub -r 5 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.10, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.15}}"

ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

判断方法：

- 如果 `angular.z=0.15` 时前轮明显转向，并且行驶半径接近 `R = L / tan(0.15)`，则 `angular.z` 是转角。
- 如果行驶半径更接近 `R = v / 0.15`，则 `angular.z` 是 yaw rate，应把 `mpc/cmd_is_yaw_rate` 改为 `true`。
- 如果方向相反，不要先调 MPC 增益，应先确认底盘坐标正方向和转向符号。

## 3. 车辆几何参数建议

你提供的 NEOS 参数：

- 外形尺寸：`0.840 x 0.626 x 0.338 m`
- 图纸轴距：`0.430 m`
- 图纸轮距/底盘宽相关尺寸：`0.548 m`
- 最小中心转弯半径：`1.34 m`
- 最小离地间隙：`0.135 m`
- 标称越障：`0.100 m`
- 最高速度：`20 km/h`
- 驱动形式：后驱、前轮转向

Ackermann 最大转角可由最小转弯半径估算：

```text
若 L = 0.430 m: phi_max ~= atan(0.430 / 1.34) = 0.31 rad = 17.8 deg
若 L = 0.535 m: phi_max ~= atan(0.535 / 1.34) = 0.38 rad = 21.8 deg
```

实车初始建议保守取：

```yaml
wheel_base: 0.430   # 若实测或厂家确认是 0.535，则改成 0.535
phimax: 0.31
kmax: 0.74          # tan(0.31) / 0.430
```

不要继续使用仿真 Hunter 的：

```yaml
wheel_base: 0.55
phimax: 0.5236
```

否则规划会生成真实 NEOS 不一定能跟踪的急弯。

如果 `base_link` 在车体几何中心，规划 footprint 初值可用：

```yaml
conpts_flat: [0.42, 0.313, 0.42, -0.313, -0.42, -0.313, -0.42, 0.313]
```

如果 `base_link` 在后轴中心或 IMU 投影点，则必须按实际 `base_link` 原点重算前后长度。不要直接套用中心对称 footprint。

### 3.1 `base_link` 是什么，怎么确定

`base_link` 是 ROS 里给车体坐标系起的名字。它不是地图坐标，也不是雷达坐标，而是固定在车身上的一个参考坐标系。一般约定为：

```text
x 轴：车头方向
y 轴：车辆左侧
z 轴：车辆上方
```

车辆在地图中的位置，通常就是 `base_link` 原点在 `map` 或 `odom` 坐标系下的位置。规划、控制、车体碰撞框、传感器安装位置，都是围绕这个车体坐标系算的。

`base_link` 的原点放在哪里不是自动算出来的，而是工程里人为约定的。常见选法有三种：

- 车体几何中心。
- 后轴中心，Ackermann 车辆常用。
- IMU 在地面上的投影点，惯导/FAST_LIO 接入时有时这样选。

当前仿真 Hunter 的 `base_link` 可以从 xacro 看出来。前轮转向 joint 在 `base_link` 下大约是：

```text
front: x ~= +0.341 m
rear : x ~= -0.208 m
```

所以仿真 Hunter 的 `base_link` 大致不在后轴中心，而是在前后轴之间偏后的位置。它不是靠规划器推导出来的，而是 URDF 里各 joint 的 `origin xyz` 定义出来的。

实车 NEOS 上，你需要自己确定一个 `base_link` 约定，并让所有配置一致。推荐两种方式：

1. 用车体几何中心作为 `base_link`。这样 footprint 最直观，NEOS 外形 `0.840 x 0.626 m` 时可先取：

```yaml
conpts_flat: [0.42, 0.313, 0.42, -0.313, -0.42, -0.313, -0.42, 0.313]
```

2. 用后轴中心作为 `base_link`。这样 Ackermann 运动学更标准，但 footprint 要改成非对称。例如 NEOS 轴距若为 `0.430 m`，整车长 `0.840 m`，且后轴到车尾距离实测为 `d_rear`，则：

```text
front_x = 0.840 - d_rear
back_x  = d_rear
half_y  = 0.626 / 2 = 0.313
```

对应：

```yaml
conpts_flat: [front_x, 0.313, front_x, -0.313, -back_x, -0.313, -back_x, 0.313]
```

关键不是选哪一个，而是全系统必须一致：

- FAST_LIO 输出的 `/Odometry` 要表示 `map -> base_link`。
- Planner 的 `conpts_flat` 要以同一个 `base_link` 为原点。
- Terrain 的 `ego_body_filter_front/back/half_width` 要以同一个 `base_link` 为原点。
- LiDAR/IMU 外参要能转换到同一个 `base_link`。
- `xnode_vehicle` 若发布 TF，frame 也会叫 `base_link`；当前建议保持 `pub_tf:false`，避免和 FAST_LIO 冲突。

实车最快的确定方法：

1. 在车体上贴一个可测量的参考点，建议先选车体几何中心或后轴中心。
2. 规定这个点就是 `base_link` 原点。
3. 用卷尺测 LiDAR、IMU、车头、车尾、左右边界相对该点的 `x/y/z`。
4. 把这些数填入 FAST_LIO 外参、body pose、planner footprint 和 terrain ego filter。
5. 在 RViz 里看 `TF`、点云、轨迹和车体框是否和真实车一致。

## 4. 已新增的实车配置文件

不要直接覆盖仿真参数。当前已经把实车文件放在各包的 `real/` 子目录下，以便和 Gazebo Hunter 配置区分：

```bash
src/FAST_LIO/config/real/neos_mid360.yaml
src/FAST_LIO/launch/real/neos_mid360_mapping.launch.py
src/uneven_mapping/terrain_analyzer/params/real/neos_registered_live.yaml
src/planner/terrain_nav/config/neos_real.yaml
src/planner/terrain_nav/launch/neos_real_terrain_nav.launch.py
src/planner/arc_planner/plan_manage/config/real/planning_neos_real.yaml
src/mpc_controller/params/real/controller_neos_real.yaml
src/planner/arc_planner/plan_manage/launch/real/neos_fastlio_nav.launch.py
src/planner/arc_planner/plan_manage/launch/real/neos_planning_control.launch.py
```

ROS2 launch 可以用简短文件名启动，不需要写 `real/` 前缀：

```bash
ros2 launch fast_lio neos_mid360_mapping.launch.py
ros2 launch plan_manage neos_planning_control.launch.py
```

这些实车配置默认：

- `use_sim_time=false`
- `planner_backend=terrain_nav`
- `cmd_topic=/cmd_vel` 或 `cmd_vel_topic=/cmd_vel`
- `odom_topic=/Odometry`
- `mpc_odom_topic=/Odometry`
- `cloud_topic=/cloud_registered`
- `mpc/cmd_is_yaw_rate=false`
- `wheel_base=0.43`
- FAST_LIO LiDAR topic 为 `/livox/lidar`
- FAST_LIO IMU topic 为 `/imu_raw`

## 5. FAST_LIO 实车注意事项

实车不能直接使用 `gazebo_hunter_mid360.yaml`，原因是其中的外参和 `body_pose_T/body_pose_R` 是 Hunter 仿真的安装关系。

实车需要创建 `neos_mid360.yaml`：

- `use_sim_time: false`
- `common/lid_topic` 和 `common/imu_topic` 改成真实 Livox 驱动发布的话题。
- `mapping/extrinsic_T`、`mapping/extrinsic_R` 使用真实 LiDAR-IMU 外参。
- `frame/body_frame` 仍建议为 `base_link`。
- `frame/body_pose_T`、`frame/body_pose_R` 使用真实 IMU body 到车辆 `base_link` 的固定变换。
- `pcd_save/pcd_save_en` 初期建议 `false`，避免长时间运行内存和磁盘压力。

FAST_LIO 的 `/Odometry` 建议作为地形分析和规划控制的统一定位源：

```text
terrain_analyzer odom <- /Odometry
terrain_nav odom      <- /Odometry
```

不要一开始把规划控制反馈换成 `xnode_vehicle` 的 `/odom`。`/odom` 是轮速积分坐标系，frame 是 `odom`；FAST_LIO 是 `map` 系。如果混用，会出现目标点、地形图、轨迹和控制反馈不在同一坐标系的问题。

`xnode_vehicle` 的 `pub_tf` 应保持 `false`，避免和 FAST_LIO 同时发布 `base_link` 相关 TF。

## 6. Terrain Analyzer 实车注意事项

当前规划 launch 默认让 terrain analyzer 订阅：

```text
cloud_topic = /cloud_registered
```

也就是说它使用 FAST_LIO 当前帧配准点云并在滚动局部网格中积累，不是直接裁剪 `/cloud_local_map`。这点在实车上仍建议保持，因为：

- `/cloud_registered` 更适合实时障碍物和近场风险。
- `/cloud_local_map` 更像建图显示，不一定适合直接当作动态避障输入。
- 如果需要利用提前建好的静态地图，应作为 prior map 或 PCD 先验单独接入，而不是简单替换实时点云。

NEOS 的地形阈值要比 Hunter 仿真更保守。当前仿真参数中：

```yaml
obstacle_min_height: 0.28
step_height_soft: 0.22
step_height_hard: 0.50
```

对 NEOS 来说偏高，因为图纸给出的越障能力只有 `0.100 m`，离地间隙 `0.135 m`。实车初值建议：

```yaml
obstacle_min_height: 0.12       # 可在 0.10~0.16 间调
obstacle_max_height: 1.45
step_height_soft: 0.06          # 可在 0.05~0.08 间调
step_height_hard: 0.12          # 可在 0.10~0.16 间调
traversable_step_free: 0.04
vertical_obstacle_inflation_m: 0.45
terrain_risk_to_hard_obstacle: true  # 初期更安全；稳定后再评估是否放宽
```

车体点过滤也要按真实车身调整。若 `base_link` 在车体中心，可先试：

```yaml
ego_clear_radius: 0.0
ego_body_filter_enable: true
ego_body_filter_front_m: 0.45
ego_body_filter_back_m: 0.45
ego_body_filter_half_width_m: 0.35
ego_body_filter_z_min_m: -0.25
ego_body_filter_z_max_m: 0.70
```

如果 `base_link` 不在车体中心，以上前后距离必须按实际安装关系重算。车体过滤过大，会把车辆附近真实障碍物清掉；过滤过小，会把车身、支架、线缆点误认为障碍物。

## 7. Planner 实车初值

从 `planning_hunter_gazebo.yaml` 复制出 `planning_neos_real.yaml`，建议初始修改：

```yaml
kmax: 0.74
vmax: 0.35
latAccmax: 0.15
lonAccmax: 0.35
accRatemax: 0.50
phimax: 0.31

safeMargin: 0.55
conpts_flat: [0.42, 0.313, 0.42, -0.313, -0.42, -0.313, -0.42, 0.313]

wheel_base: 0.43
mapRes: 0.2
mapX: 25.0
mapY: 25.0

front_end/allow_reverse: false
front_end/use_footprint_collision_check: true
front_end/footprint_safe_margin_scale: 0.60
front_end/soft_clearance: 0.65
```

说明：

- 实车初期先禁用倒车，等前进跟踪和停车稳定后再打开。
- `vmax` 不应按 NEOS 最高速度 `20 km/h` 设置。自动规划初期建议 `0.25~0.35 m/s`。
- `safeMargin` 要覆盖定位漂移、控制滞后、点云稀疏、底盘宽度和真实障碍误差。

## 8. 旧 MPC 回退链路实车初值

当前默认规划控制链路已经切到 `terrain_nav`。下面参数仅用于手动回退到 `planner_backend:=arc mpc:=true` 时参考。

从 `controller_hunter_gazebo.yaml` 复制出 `controller_neos_real.yaml`，建议初始修改：

```yaml
mpc/cmd_is_yaw_rate: false

mpc/dt: 0.10
mpc/predict_steps: 10
mpc/max_steer: 0.31
mpc/max_dsteer: 0.80
mpc/max_speed: 0.35
mpc/min_speed: 0.00
mpc/reverse_speed_cap: 0.08
mpc/max_accel: 0.35
mpc/wheel_base: 0.43
mpc/delay_num: 1

mpc/force_reverse_follow: false
mpc/cmd_speed_scale: 1.00
mpc/turn_speed_steer_gain: 0.55
mpc/turn_speed_min_cap: 0.10
```

调试顺序：

1. 先只前进，不允许倒车：`min_speed: 0.00`，`front_end/allow_reverse: false`。
2. 确认转向语义和方向正确后，再允许小速度倒车：`min_speed: -0.08`。
3. 如果车辆跟踪有明显延迟，再增加 `mpc/delay_num` 到 `2` 或 `3`。
4. 如果转向抖动，先降低 `max_dsteer`，再调 `matrix_rd`，不要直接提高速度。

注意：`mpc/cmd_speed_scale` 在代码里会被限制到 `0.5~3.0`，实车限速应优先通过 `mpc/max_speed` 和 planner `vmax` 做。

## 9. 实车启动顺序

### 9.1 编译和 source

```bash
cd /home/z/workspace_zyz/seb_clean
unset PYTHONHOME PYTHONPATH
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

驱动工作空间：

```bash
source /opt/ros/humble/setup.bash
source /home/z/workspace_zyz/driver/hexman/sdk_neos_ws/install/setup.bash
```

### 9.2 启动 NEOS 底盘驱动

```bash
source /opt/ros/humble/setup.bash
source /home/z/workspace_zyz/driver/hexman/sdk_neos_ws/install/setup.bash
ros2 launch xpkg_vehicle xnode_vehicle_test.launch.py
```

现有 `xnode_vehicle_test.launch.py` 没有声明 launch 参数。如果要改 `rate_x`、`rate_az`、`mode_can_lock`、`pub_tf`，需要修改该 launch 或新建一个 NEOS 实车专用 launch。

建议初始：

```yaml
mode_can_lock: true
pub_tf: false
calc_speed: false
rate_x: 1.0
rate_az: 1.0
```

如果第一次空载测试担心速度过大，可以临时把 `rate_x` 和 `rate_az` 降到 `0.5`。但闭环规划测试时要恢复为 `1.0`，否则 MPC 以为自己发出了 `0.31 rad` 转角，底盘实际只收到 `0.155 rad`，会造成系统性跟踪不足。

确认驱动状态：

```bash
ros2 topic hz /odom
ros2 topic echo /xtopic_vehicle/device_state_json
ros2 topic info -v /cmd_vel
```

`device_state_json` 中重点看：

- `work_mode` 是否为 `2`，即 CAN 模式。
- `err_stop`、`err_motor`、`err_driver` 是否为正常。
- `battery_vol` 是否正常。
- `speed_x`、`speed_r` 是否随低速命令变化。

### 9.3 启动 Livox 驱动

启动真实 MID360 ROS2 驱动后确认：

```bash
ros2 topic hz /livox/lidar
ros2 topic hz /imu_raw
ros2 topic info -v /livox/lidar
ros2 topic info -v /imu_raw
```

如果真实话题名不是 `/livox/lidar`、`/imu_raw`，要同步修改 `neos_mid360.yaml`。

### 9.4 启动 FAST_LIO

```bash
cd /home/z/workspace_zyz/seb_clean
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch fast_lio mapping_hunter_sim.launch.py \
  use_sim_time:=false \
  config_path:=/home/z/workspace_zyz/seb_clean/src/FAST_LIO/config \
  config_file:=real/neos_mid360.yaml \
  rviz:=true
```

更推荐使用已经新增的实车 launch：

```bash
ros2 launch fast_lio neos_mid360_mapping.launch.py
```

该 launch 默认使用 `src/FAST_LIO/config/real/neos_mid360.yaml`，并且默认 `use_sim_time=false`。

确认：

```bash
ros2 topic hz /Odometry
ros2 topic hz /cloud_registered
ros2 topic hz /cloud_local_map
ros2 run tf2_ros tf2_echo map base_link
```

手推或遥控低速行驶时，FAST_LIO 地图应稳定，车辆位姿方向应和实际一致。

### 9.5 启动地形、规划、控制

推荐分两个终端启动。第一个终端启动 FAST_LIO 和地形分析：

```bash
cd /home/z/workspace_zyz/seb_clean
./run_real_mapping.sh
```

第二个终端启动当前 `terrain_nav` 规划控制和可选 NEOS 底盘驱动：

```bash
cd /home/z/workspace_zyz/seb_clean
./run_real_nav.sh
```

`run_real_nav.sh` 默认使用 `planner_backend:=terrain_nav`，不会启动旧 `plan_node + mpc_controller`。如果只想看 RViz 和规划，不接底盘驱动，可加：

```bash
./run_real_nav.sh vehicle_driver:=none
```

确认地形层：

```bash
ros2 topic hz /car_0_terrain_analyzer/fused_map
ros2 topic hz /car_0_terrain_analyzer/sdf_map
ros2 topic hz /car_0_terrain_analyzer/risk_cloud
ros2 topic hz /car_0_terrain_analyzer/obstacle_cloud
```

确认目标点能生成轨迹：

```bash
ros2 topic hz /terrain_nav/path
ros2 topic hz /reference_path
ros2 topic hz /visualization/terrain_guidance_path
```

旧 `plan_manage + mpc_controller` 链路仅作为回退入口：

```bash
./run_real_nav.sh planner_backend:=arc mpc:=true
```

## 10. 分阶段上车测试流程

### 阶段 A：底盘驱动独立测试

目标：确认 `/cmd_vel` 控制语义、正方向、急停、CAN 模式。

操作：

1. 车轮离地或空旷低速。
2. 只启动 `xnode_vehicle`。
3. 发布小速度直行、停止、小转角前进、停止。
4. 记录 `speed_x`、`speed_r` 和实际运动方向。

通过标准：

- 正 `linear.x` 是前进。
- 正 `angular.z` 的转向方向和 ROS 坐标约定一致。
- 停止命令能立即停止。
- 遥控器或急停能覆盖自动控制。

### 阶段 B：FAST_LIO 独立测试

目标：确认外参、时间戳、TF 和地图稳定。

操作：

1. 启动 Livox + FAST_LIO。
2. 不启动规划控制。
3. 手推或遥控低速绕场地一圈。

通过标准：

- `/Odometry` 连续无跳变。
- `/cloud_registered` 和 `/cloud_local_map` 与真实环境对齐。
- `map -> base_link` 方向正确。
- 静止时位姿不明显漂移。

### 阶段 C：Terrain only

目标：确认障碍物高度、台阶、车体过滤和风险层。

操作：

```bash
ros2 launch plan_manage neos_fastlio_nav.launch.py terrain:=true planner:=false mpc:=false rviz:=true
```

通过标准：

- 车辆附近车身点不会被识别为障碍。
- 真实 10 cm 左右障碍会进入 obstacle/risk。
- 地面平坦区域 risk 低。
- 车体周边不会出现大面积无效空洞，除非确实是 LiDAR 盲区或未观测区域。

### 阶段 D：Planner only

目标：确认目标点、ESDF、risk 和 footprint 约束工作。

操作：

1. `planner_backend:=terrain_nav`。
2. 在 RViz 里给近距离目标点，先 `1~2 m`。
3. 看 `/terrain_nav/path`、`/reference_path` 和 `/visualization/terrain_guidance_path`。

通过标准：

- 轨迹不穿障碍。
- 曲率不超过 NEOS 转向能力。
- 不在狭窄场景中频繁生成倒车轨迹。

### 阶段 E：闭环低速

目标：确认 `terrain_nav` 闭环控制能低速稳定跟踪。

操作：

1. `vmax/max_speed <= 0.35 m/s`。
2. `allow_reverse=false`。
3. 目标点距离从 `1 m` 开始。
4. 现场必须有人拿急停或遥控接管。

通过标准：

- 起步不猛冲。
- 转向不反向。
- 终点附近能减速停车。
- 停止规划或关闭 `terrain_nav_node` 后 `/cmd_vel` 归零。

## 11. 常见问题和处理

### 11.1 实车节点不动或时间不更新

检查是否误用了仿真时间：

```bash
ros2 param get /terrain_nav_node use_sim_time
ros2 param get /car_0_terrain_analyzer use_sim_time
```

实车必须是：

```text
use_sim_time = false
```

### 11.2 车辆转弯半径明显比规划大

优先检查：

- `mpc/max_steer` 是否大于真实最大转角。
- `mpc/wheel_base` 是否使用了错误轴距。
- `xnode_vehicle` 的 `rate_az` 是否小于 `1.0`。
- `/cmd_vel.angular.z` 到底是转角还是 yaw rate。

### 11.3 车辆跟踪方向反了

不要先调 PID/MPC 权重。先检查：

- `base_link` x 轴是否指向车头。
- FAST_LIO `body_pose_R` 是否正确。
- `/cmd_vel.angular.z` 正方向是否和车辆实际左转/右转一致。
- RViz 中轨迹坐标系是否是 `map`。

### 11.4 地形层把近处障碍清掉

检查：

- `ego_body_filter_front/back/half_width` 是否过大。
- `ego_clear_radius` 是否非零且过大。
- `obstacle_min_height` 是否高于实际障碍。
- LiDAR 是否确实扫到该障碍。

实车上近场空洞不一定是建图问题。当前 terrain 输入是 `/cloud_registered`，它依赖实时观测；如果障碍处在 LiDAR 近场盲区、被车体遮挡、或被车体过滤区域覆盖，就不会稳定进入 risk/SDF。

### 11.5 使用 `/cloud_local_map` 是否能解决盲区

不建议直接把 terrain 输入改成 `/cloud_local_map` 作为第一方案。原因：

- `/cloud_local_map` 更适合建图可视化，可能包含历史静态点。
- 动态障碍物可能滞留或更新不及时。
- 局部地图的点云密度和 FAST_LIO 内部维护策略会影响地形分析稳定性。

更合理的方式是：

1. 用 `/cloud_registered` 做实时障碍。
2. 对已知静态场地使用 PCD prior 或 map prior。
3. 在 terrain analyzer 内明确区分 prior map 和 live obstacle。

## 12. 上车前安全清单

每次闭环测试前确认：

- 急停可用。
- 遥控器可接管。
- `mode_can_lock` 行为符合预期。
- 没有其他节点同时发布 `/cmd_vel`。
- `use_sim_time=false`。
- `mpc/max_speed <= 0.35` 初期测试。
- `front_end/allow_reverse=false` 初期测试。
- `xnode_vehicle pub_tf=false`。
- FAST_LIO `map -> base_link` 正常。
- RViz 目标点在 `map` frame。
- `/cmd_vel` 停止命令已验证有效。

## 13. 推荐的工程化改造

当前可以用参数传入方式跑实车，但长期建议把仿真和实车彻底拆开：

1. `neos_planning_control.launch.py` 默认使用 `terrain_nav`，旧 `arc` 后端仅保留为回退。
2. `neos_registered_live.yaml` 和 `terrain_nav/config/neos_real.yaml` 不复用 Hunter 仿真阈值。
3. `neos_mid360.yaml` 只保留真实外参。
5. 为 `xnode_vehicle` 新建无 RViz 的 bringup launch，并显式写清 `mode_can_lock`、`pub_tf`、`rate_x`、`rate_az`。
6. 增加 `/cmd_vel` 安全门或 watchdog，避免规划/控制节点异常退出后底盘继续执行上一条速度命令。

实车优先目标不是跑快，而是确保坐标、转向、限幅、地形风险和停车行为全部可信。确认这些基础项后，再逐步提高速度、打开倒车、缩小安全距离。
