# 给下一个 AI 的上下文备忘录 (Context Compression)

**当前系统状态**:
用户正在调试一个基于 ROS2 的复杂地形 UGV 自动驾驶系统。系统融合了 FAST-LIO、Terrain Analyzer（风险图）、Kino-A*、L-BFGS 多项式优化和 CasADi NMPC。

**目前遗留的核心问题 (Current Open Issues)**:
1. **严重偏离与起点跳变 (Track Mismatch)**: 重规划时，新参考轨迹的起点位置和朝向经常与车辆的真实物理状态有很大偏差。导致车辆拼命纠偏，出现“跟不上”的现象。
2. **终点前依然无法到达且轨迹凌乱**: 在到达终点附近时，系统不仅不规划最后一段，而且上一段轨迹往往是极度扭曲弯折的。
3. **MPC 静态崩溃 (`fail(551) | v=0.00`)**: 当车辆静止起步或低速时，CasADi 优化器陷入死锁，抛出最大迭代次数超限错误。
4. **跟随滞后与速度不足**: 尽管释放了底层参数，但规划出来的路径与 MPC 的响应依然存在脱节。

**此前已尝试过的修改**:
- 已尝试在 `replan_fsm.cpp` 中通过航向阈值过滤不合理的预测起点，但目前的 fallback 运动学前推方案依然会导致位置/朝向跳变。
- 在 `kino_astar.cpp` 中放宽了近距离目标点的直接连线限制（`len < 0.40 && yaw_span < 0.60`），但这可能导致了终点前轨迹的极度扭曲（强行折角）。
- 已将仿真器 (`multi_sim.launch.py`) 的物理限制拉满 (`2.5m/s`) 并同步了 MPC 参数，但脱节问题依然存在。

---

**给下一个 AI 的行动指令 (Action Items for the Next Prompt)**:
当你接手此会话时，请直接针对以上问题进行**协同修改**，切忌顾此失彼：

1. **重构起步状态预测 (Fix Track Mismatch)**：在 `replan_fsm.cpp` 中，彻底修改 `launchAsyncReplan()` 的 `start_state` 获取逻辑。放弃简单粗暴的线性前推，必须保证新规划的起点 100% 严丝合缝地贴合当前车身的实际物理位置和朝向！
2. **根除 MPC 起步崩溃 (Fix MPC fail 551)**：在 `mpc.cpp` 中排查 `v=0.00` 时的 `xref` 状态，修复初始猜测矩阵 `X_sol` 和 `U_sol` 在极低速时的奇异性/不兼容约束。
3. **修复终点扭曲与不到达 (Fix Goal Reaching)**：检查 `kino_astar.cpp` 的 `is_shot_sucess`，如果是由于目标朝向偏差过大导致的扭曲，应引导 A* 倒车微调而不是硬切。并核实 `plan_manage.cpp` 中终点速度和状态机抵达阈值的匹配情况。