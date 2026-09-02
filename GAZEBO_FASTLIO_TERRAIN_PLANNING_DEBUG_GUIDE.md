# 规划与控制系统调参及 Debug 手册

## 1. 典型故障与病因排查

### 1.1 MPC 报错 `fail(201)/351/501: Maximum_Iterations_Exceeded` 且车辆偏离或停死
- **病因**：NMPC 收到了物理上“不可能实现”的参考轨迹。通常是因为规划器在极短距离内给出了极大的速度突变（例如从 1.5m/s 瞬间要求变到 0m/s），或者轨迹产生了严重的波浪状畸变（扭曲成麻花）。
  *补充：如果报错时 `v=0.00`，通常是因为起步状态的 Initial Guess 不佳，或者第一段轨迹的航向角与车身当前航向角有巨大的断层跳变（Track Mismatch）。*
- **解决/排查点**：
  1. 检查 `plan_manage.cpp` 中的末端速度分配。绝对不能在距离终点还有 0.4 米时强制截断速度 `end_vel_mag = 0.0`。
  2. 检查 `kino_astar.cpp` 里的 `is_shot_sucess`。如果在位姿不正的情况下强行直线连接终点，送入后端会被 L-BFGS 扭曲。
  3. 检查 `mpc.cpp` 中的 `U_sol` 初始猜测，避免在静止起步时给予过大或方向错误的初始速度猜测。

### 1.2 车辆在终点前 0.3~0.5 米处停下，无法生成最后一段轨迹
- **病因**：
  1. **FSM 状态机提前下班**：`replan_fsm.cpp` 里的抵达阈值 `raw_goal_dist < 0.35` 设得太大。
  2. **MPC 提前拉手刹**：`mpc.cpp` 里的 `traj_analyzer.at_goal` 逻辑中，由于几何投影判断到达，直接锁死了求解器并输出 `v=0`。
  3. **A* 跨步越界**：Kino-A* 步长约 0.4m，无法恰好落入过于严苛的终点容忍区，导致找不到路。
- **解决/排查点**：在 `mpc.cpp` 移除强制 `v=0` 逻辑，让求解器自行跟随到目标；在 A* 中放宽终点前 1 米的曲率限制；开启倒车允许 (`allow_reverse: True`)。

### 1.3 轨迹倾向于贴着障碍物规划，且产生多余的“弯弯绕绕”
- **病因**：
  1. **风险场/膨胀不够**：A* 认为离墙 0.2 米就是绝对安全，自然会抄近道贴墙走。
  2. **转向惩罚太低**：绕弯的 Cost 小于走直线的 Cost。
- **解决/排查点**：
  1. 在 `multi_sim.launch.py` / `terrain_analyzer` 中，大幅增加 `tree_risk_radius_m`（例如 1.5m）和 `safeMargin`。
  2. 在 `kino_astar.cpp` 的 `tmp_g_score` 计算中，对踩入风险区的行为施加极重的惩罚（例如乘以 35 倍的罚金）。
  3. 提高 `traj_steer_penalty`，逼迫规划器走直线。

### 1.4 车辆移速非常缓慢（老头乐）
- **病因**：后端多项式优化器 L-BFGS 为了压低由于高速度带来的加加速度 (Jerk) 代价，主动拉长了轨迹的总时间。
- **解决/排查点**：在 `plan_manage.cpp` 中适当降低时间权重 `config_->wei_time_`（例如降至 5.0 ~ 20.0），逼迫优化器压缩时间、提高下发的参考速度。

## 2. 核心参数速查字典

| 文件名 | 参数名 | 作用描述 | 推荐值/调试方向 |
| :--- | :--- | :--- | :--- |
| `multi_sim.launch.py` | `tree_risk_radius_m` | 地形分析器的软风险场扩散半径 | 增大(1.5+)可远离障碍物 |
| `multi_sim.launch.py` | `safeMargin` | 规划器全局膨胀半径 | 0.55 ~ 0.85 之间 |
| `multi_sim.launch.py` | `mpc/max_speed` | 底层硬件允许的最大线速度 | 2.0 |
| `multi_sim.launch.py` | `front_end/traj_steer_penalty` | 转向惩罚，越高轨迹越直 | 5.0 左右 |
| `plan_manage.cpp` | `config_->wei_time_` | 优化器时间代价权重 | 调低(5~20)提速，太低会导致波浪 |
| `kino_astar.cpp` | `risk_free_threshold_` | 认为绝对安全的风险上限 | 0.3~0.4 |
| `replan_fsm.cpp` | 抵达容差 (`raw_goal_dist < X`) | 判定任务完成的距离 | 收紧至 0.15 提高精度 |