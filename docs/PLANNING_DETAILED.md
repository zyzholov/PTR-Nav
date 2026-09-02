# ARC-Planner 路径规划详细流程与原理

## 一、总体架构

本工程采用经典的 **前端-后端** 分离式规划架构，但针对 **Ackermann 车辆（非完整约束）** 做了深度定制。整体流水线为:

```
ReplanFSM (状态机)
    │
    ▼
PlanManager::process()
    │
    ├─ 1. Kino A* 前端搜索 → 粗糙路径 (离散状态序列)
    │
    ├─ 2. 提取运动学可行采样点 → 分段 (前进/后退)
    │
    ├─ 3. 时间分配 (梯形速度剖面)
    │
    ├─ 4. MINCO 轨迹初始化 (弧长参数化)
    │
    ├─ 5. ALM + L-BFGS 后端优化 → 平滑轨迹
    │
    ├─ 6. 安全性验证 (SDF + Risk 检查)
    │
    └─ 7. 发布多项式轨迹 → MPC 控制器
```

---

## 二、前端: Kino A* 运动学搜索

### 2.1 与常规 A* 的区别

常规 A*（如 FAST-Planner 的前端）在 **3D 位置空间** `(x, y, z)` 中搜索，使用欧氏距离作为代价，扩展邻居时采用简单的 6-连通或 26-连通网格。

**Kino A*** (Kinodynamic A*) 的核心区别在于:

| 特性 | 常规 A* | Kino A* |
|------|---------|---------|
| 状态空间 | `(x, y, z)` | `(x, y, θ)` — 包含航向角 |
| 扩展方式 | 网格邻居 | **运动学传播** — 使用车辆模型 |
| 可行性 | 无保证 | 生成的路径满足运动学约束 |
| 代价函数 | 欧氏距离 | 距离 + 转向 + 换挡 + 方向惩罚 |

### 2.2 车辆运动学模型 (Bicycle Model)

状态为 `(x, y, θ)`，控制输入为 `(ψ, s)`，其中 `ψ` 是前轮转向角，`s` 是弧长。

**状态转移方程:**

当 `|ψ| ≥ 0.01`（转弯）:
```
κ = tan(ψ) / L                    // 曲率, L 为轴距

x'  = x + (1/κ) · [sin(θ + κ·s) - sin(θ)]
y'  = y - (1/κ) · [cos(θ + κ·s) - cos(θ)]
θ'  = θ + κ·s
```

当 `|ψ| < 0.01`（直行）:
```
x'  = x + s · cos(θ)
y'  = y + s · sin(θ)
θ'  = θ
```

这就是 Dubins 曲线中使用的自行车模型精确弧长传播，而非简单的欧氏步进。

### 2.3 搜索空间与离散化

搜索在一个 **3D 网格** `(x_idx, y_idx, yaw_idx)` 中进行:

- **位置分辨率**: `resolution = 0.2m`（与地形图分辨率一致）
- **航向分辨率**: `yaw_resolution = 0.18 rad`（约 10.3°）
- **弧长离散**: `step_arc = 0.36m`，步长 `0.5 × step_arc`
- **转向角离散**: `steer_res = 0.20`，步长 `steer_res × max_steer`

每次扩展生成的控制动作组合:
```
arc ∈ {-step_arc, -0.5·step_arc, +0.5·step_arc, +step_arc}
steer ∈ {-max_steer, ..., 0, ..., +max_steer}
```

### 2.4 代价值函数

从当前节点 `cur` 扩展到邻居 `pro` 的代价为:

```
g(pro) = g(cur) + Δg
```

其中 `Δg` 由以下项组成:

```
Δg = |s| · penalty_dir                     // 方向代价 (前进=1.0, 后退=2.0)
   + gear_switch_penalty · δ(dir切换)       // 换挡代价 (14.0)
   + steer_penalty · |ψ| · |s|              // 转向角代价 (0.75)
   + 0.22 · (|ψ|/ψ_max)² · |s|             // 二次转向惩罚
   + steer_change_penalty · |ψ - ψ_prev|    // 转向角变化代价 (0.85)
```

**启发函数:**
```
h(x, x_goal) = λ · √[(x-x_g)² + (y-y_g)²]
```
其中 `λ = 1.0`（`lambda_heu_`），使用欧氏距离作为下界估计。

**设计原理:**
- 后退代价 (2.0) 是前进 (1.0) 的 2 倍 → 鼓励前进
- 换挡代价 (14.0) 很高 → 避免频繁前进/后退切换
- 转向角变化惩罚 → 鼓励平滑路径

### 2.5 碰撞检测

使用 SDF (Signed Distance Field) 进行碰撞检测:

```
isCollided(state):
    d = SDF(state.x, state.y)
    return (d ≤ safe_dis)    // safe_dis = 0.42m
```

沿每条弧均匀采样 `check_num = 4` 个中间点进行检测:

```
for k = 1 to check_num:
    s_partial = s · (k / check_num)
    state_partial = stateTransit(cur_state, ψ, s_partial)
    if isCollided(state_partial): return COLLIDED
```

### 2.6 One-Shot 直达轨迹

当搜索节点距目标 ≤ 1.5m 时，尝试直接连接:

```
is_shot_success(state1, state2):
    // 检查航向偏差
    line_yaw = atan2(dy, dx)
    start_gap = min(|θ1 - line_yaw|, |θ1 - line_yaw - π|)
    end_gap   = min(|θ2 - line_yaw|, |θ2 - line_yaw - π|)

    // 检查隐含曲率
    implied_cur = max(2·sin(start_gap)/L, 2·sin(end_gap)/L, |Δθ|/L)

    if start_gap > 1.15 rad OR implied_cur > 1.2·κ_max:
        REJECT  // 航向偏差太大或曲率超限

    // 检查碰撞
    for each interpolated point:
        if isCollided(point): REJECT

    return SUCCESS
```

这类似于 EGO-Planner 中的 "kinodynamic shortcut"，但针对 Ackermann 车辆增加了曲率可行性检查。

### 2.7 梯形速度剖面 (时间分配)

前端搜索完成后，需要为路径分配时间。采用 **梯形速度剖面**:

```
对于每个换挡段 (从 gear_switch[i] 到 gear_switch[i+1]):

    L = 该段总弧长
    v_start, v_end = 起止速度
    v_max, a_max = 最大速度和加速度

    // 临界长度: 能否加速到 v_max 再减速回来
    L_crit = (v_max² - v_start²)/(2·a_max) + (v_max² - v_end²)/(2·a_max)

    if L ≥ L_crit:
        // 梯形剖面: 加速 → 匀速 → 减速
        t_accel = (v_max - v_start) / a_max
        t_cruise = (L - L_crit) / v_max
        t_decel = (v_max - v_end) / a_max
        T = t_accel + t_cruise + t_decel
    else:
        // 三角剖面: 加速到峰值 → 减速
        v_peak = √[v_start²/2 + v_end²/2 + a_max·L]
        T = (v_peak - v_start)/a_max + (v_peak - v_end)/a_max
```

---

## 三、轨迹表示: 双 MINCO 弧长参数化

### 3.1 与常规轨迹表示的区别

| 框架 | 轨迹表示 | 参数化方式 |
|------|----------|------------|
| **FAST-Planner** | MINCO (位置 vs 时间) | `σ(t) = Σ cᵢ·tⁱ` |
| **EGO-Planner** | MINCO (位置 vs 时间) | 同上 |
| **本工程** | **双 MINCO** (位置 vs 弧长 + 时间 vs 弧长) | 几何与时间解耦 |

### 3.2 五次多项式基

每段多项式为 5 阶（6 个系数/维度）:

```
σ(τ) = c₀ + c₁·τ + c₂·τ² + c₃·τ³ + c₄·τ⁴ + c₅·τ⁵
```

其中 `τ` 是段内局部参数（归一化时间或弧长）。

导数解析计算:
```
σ'(τ)  = c₁ + 2c₂·τ + 3c₃·τ² + 4c₄·τ³ + 5c₅·τ⁴
σ''(τ) = 2c₂ + 6c₃·τ + 12c₄·τ² + 20c₅·τ³
σ'''(τ)= 6c₃ + 24c₄·τ + 60c₅·τ²
```

### 3.3 MINCO_S3NU: 非均匀时间 MINCO

给定:
- 内部路点 `inPs`（位置在内部节点处）
- 段时长 `ts`（每段一个）
- 首尾状态（位置、速度、加速度）

构造带状线性系统 `A·c = b`，通过 LU 分解 O(N) 求解。系统约束:
- 位置连续: `σᵢ(Tᵢ) = σᵢ₊₁(0)`
- 速度连续: `σᵢ'(Tᵢ) = σᵢ₊₁'(0)`
- 加速度连续: `σᵢ''(Tᵢ) = σᵢ₊₁''(0)`
- jerk 连续: `σᵢ'''(Tᵢ) = σᵢ₊₁'''(0)`
- 首尾 PVA 条件指定

### 3.4 双轨迹解耦

**核心思想: 将几何形状与运动时序分离**

同时优化两条 MINCO 轨迹:

1. **位置轨迹** `posTraj`: `MINCO_S3NU<2>` — 2D 位置作为 **弧长 s** 的函数
   ```
   p(s) = (x(s), y(s))
   ```

2. **时间映射轨迹** `timeTraj`: `MINCO_S3NU<1>` — 1D 弧长作为 **时间 t** 的函数
   ```
   s(t) = 弧长随时间的变化
   ```

**实际状态恢复:**
```
位置:     p(t) = posTraj(s(t))
速度:     v(t) = dp/ds · ds/dt
加速度:   a(t) = d²p/ds² · (ds/dt)² + dp/ds · d²s/dt²
曲率:     κ(t) = (vₓ·aᵧ - aₓ·vᵧ) / |v|³
```

**为什么用弧长参数化？**

对于 Ackermann 车辆，速度变化不应改变路径形状。弧长参数化自然保证了:
- 几何形状与速度解耦
- 曲率计算更稳定（不受速度波动影响）
- 低速/停车时轨迹定义仍然良好

---

## 四、后端: ALM + L-BFGS 轨迹优化

### 4.1 与常规优化方法的区别

| 框架 | 优化方法 | 约束处理 |
|------|----------|----------|
| **FAST-Planner** | 无约束 L-BFGS | 罚函数法 |
| **EGO-Planner** | 无约束 L-BFGS | 罚函数法 + ESDF |
| **本工程** | **ALM + L-BFGS** | 增广拉格朗日法（更严格的约束满足） |

### 4.2 优化变量

决策变量 `x` 包括:

```
x = [P₁, P₂, ..., Pₘ,      // 各段内部路点 (2D)
     Vdt₁, Vdt₂, ...,       // 虚拟时间参数
     G₁, G₂, ...,            // 换挡点位置 (2D)
     α₁, α₂, ...,            // 换挡点航向角
     Vs₁, Vs₂, ...]          // 弧长缩放因子
```

### 4.3 虚拟-实时间映射

为了在无约束优化中保证时间正值:

```
if VT > 0:   RT = (0.5·VT + 1)·VT + 1 + ε
if VT ≤ 0:   RT = 1/((0.5·VT - 1)·VT + 1) + ε
```

其中 `ε = 0.05`（`gslar_`）。这是一个光滑的 C² 双射: ℝ → (ε, +∞)。

### 4.4 代价函数总览

总代价 = 平滑性 + 时间惩罚 + 约束惩罚 (via ALM)

```
J = J_smooth + J_time + J_alm
```

#### 4.4.1 平滑性代价 (Jerk 最小化)

```
J_smooth = Σᵢ Σⱼ w · Δt · ||realJerk||²
```

其中 realJerk 考虑了弧长参数化:
```
realJerk = d³s/dt³ · vel + acc · d²s/dt² · ds + 2·ds·acc·d²s/dt² + ds³ · jerk_poly
```

权重: 前进段 `w = 0.1`，后退段 `w = backW (6.0)` — 后退更需要平滑。

#### 4.4.2 时间惩罚

```
J_time = Σᵢ k · wei_time · Δtᵢ · N_pieces
```

`k = 1`（前进）或 `k = backW`（后退），鼓励更短的执行时间。

#### 4.4.3 约束项 (ALM 处理)

以下约束通过 **增广拉格朗日法 (ALM)** 处理:

**(a) 最小速度约束 (非退化)**
```
g₁: miniS² - ||vel||² ≤ 0
```
确保速度不退化为零（否则曲率未定义）。

**(b) 曲率约束**
```
κ = (vₓ·aᵧ - aₓ·vᵧ) / |v|³ · inv_cos_ξ · inv_cos_φₓ

g₂: 100·(κ² - κ_max²) ≤ 0
```

其中 `inv_cos_ξ` 和 `inv_cos_φₓ` 是地形坡度修正因子。

**(c) 弧长单调性约束**
```
g₃: -ds ≤ 0    (即 ds/dt ≥ 0)
```
保证弧长参数始终正向推进。

**(d) 速度约束**
```
vₓ = ds · |vel| · inv_cos_φₓ

g₄: 100·(vₓ² - v_max²) ≤ 0
```

**(e) 纵向加速度约束**
```
ax_body = (acc·vel / |vel|) · inv_cos_φₓ + g · sin_φₓ

g₅: ax_body² - a_lon_max² ≤ 0
```

**(f) 横向加速度约束**
```
ay_body = (vel · acc_⊥ / |vel|) · ds² · inv_cos_φᵧ + g · sin_φᵧ

g₆: ay_body² - a_lat_max² ≤ 0
```

**(g) ESDF 避障约束**

对车辆 footprint 的每个角点:
```
p_abs = pos + R(θ) · p_body

d = SDF(p_abs)
violation = safe_margin - d

if violation > 0:
    J_esdf += esdfWei · smoothedL1(violation)
```

其中旋转矩阵 `R(θ)` 使用速度方向对齐:
```
R = [vₓ, -vᵧ; vᵧ, vₓ] / |vel|
```

**(h) 地形风险约束**
```
g₈: (risk - 0.5) · 0.1 ≤ 0
```

### 4.5 地形坡度积分 (TPM - Terrain Property Map)

这是本工程独有的特性。`getTPMWithGrad()` 从地形图读取法向量分量 `(zbx, zby)` 并计算:

```
zb = (zbx, zby)           // 地形法向量的 x, y 分量
c = √(1 - zbx² - zby²)    // 法向量的 z 分量

// 车辆航向与地形的耦合
t = cos(θ)·zbx + sin(θ)·zby    // 纵向坡度分量
s = -sin(θ)·zbx + cos(θ)·zby   // 横向坡度分量

// 纵向坡度因子
inv_cos_φₓ = 1/√(1 - t²)
sin_φₓ = -c·t / √(1 - t²)

// 横向坡度因子
inv_cos_φᵧ = √(1 - t²) / c
sin_φᵧ = s / √(1 - t²)

// 航向-坡度耦合因子
cos_ξ = c
inv_cos_ξ = 1/c
```

这些因子修正了速度、加速度和曲率的计算，使其考虑了地形倾斜的影响。

### 4.6 ALM (增广拉格朗日法) 更新

对于每个不等式约束 `gᵢ(x) ≤ 0`:

```
// 增广拉格朗日函数
L(x, λ, μ, ρ) = f(x) + Σᵢ [μᵢ·gᵢ(x) + (ρ/2)·max(0, gᵢ(x) + μᵢ/ρ)²]

// 对偶变量更新 (外循环)
μᵢ = max(0, μᵢ + ρ · scaling · gᵢ(x))

// 罚参数更新
ρ = min((1 + γ) · ρ, ρ_max)

// 收敛判定
max(|gᵢ(x)|) < ε    → 收敛
```

**为什么用 ALM 而不是简单罚函数？**

| 方法 | 优点 | 缺点 |
|------|------|------|
| 罚函数法 | 实现简单 | 罚参数太大导致病态，太小约束不满足 |
| ALM | 约束满足更严格，数值稳定性好 | 实现复杂，需要外循环 |

ALM 通过显式维护拉格朗日乘子 `μ`，可以在有限的罚参数下精确满足约束。

### 4.7 L-BFGS 内层求解

ALM 的每轮外循环中，使用 L-BFGS 求解无约束子问题:

```
x_{k+1} = x_k - α_k · H_k · ∇L(x_k)
```

其中 `H_k` 是 L-BFGS 近似 Hessian 矩阵。

**时间预算机制:**
```
if (wall_clock - lbfgs_start) > lbfgs_time_budget:
    early_exit    // 动态时间预算 (0.8-2.0s)
```

---

## 五、与 FAST-Planner / EGO-Planner 的详细对比

### 5.1 架构对比

```
FAST-Planner:
    Kinodynamic A* (3D) → B样条优化 → 坐标下降 (无约束 L-BFGS)
    状态: (x, y, z)     轨迹: B-spline    优化: 罚函数

EGO-Planner:
    Kinodynamic A* (3D) → MINCO优化 → 无约束 L-BFGS
    状态: (x, y, z)     轨迹: MINCO       优化: 罚函数 + ESDF

本工程 (ARC-Planner):
    Kino A* (SE2) → 双MINCO优化 → ALM + L-BFGS
    状态: (x, y, θ)    轨迹: 双MINCO      优化: 增广拉格朗日
```

### 5.2 前端对比

| 特性 | FAST-Planner | EGO-Planner | ARC-Planner |
|------|--------------|-------------|-------------|
| 搜索空间 | (x,y,z,vel) 4D | (x,y,z) 3D | **(x,y,yaw) 3D** |
| 运动模型 | 点质量模型 | 点质量模型 | **自行车模型** |
| 扩展方式 | 26-连通 + 动力学 | 26-连通 | **弧长+转向角** |
| 换挡支持 | 无 | 无 | **有 (多段)** |
| 碰撞检测 | 3D ESDF | 3D ESDF | **2D SDF** |
| 曲率检查 | 无 | 无 | **有** |

### 5.3 后端对比

| 特性 | FAST-Planner | EGO-Planner | ARC-Planner |
|------|--------------|-------------|-------------|
| 轨迹表示 | B-spline | MINCO (时间) | **MINCO (弧长+时间)** |
| 优化方法 | 坐标下降 + L-BFGS | 无约束 L-BFGS | **ALM + L-BFGS** |
| 约束类型 | 罚函数 | 罚函数 | **增广拉格朗日** |
| 避障方式 | 3D ESDF | 3D ESDF + 罚函数 | **2D ESDF + Footprint** |
| 地形感知 | 无 | 无 | **坡度/曲率/风险** |
| 曲率约束 | 无 | 无 | **有 (κ_max)** |
| 加速度分解 | 无 | 无 | **纵向/横向分离** |

### 5.4 独有特性

**1. 弧长参数化 (vs 时间参数化)**

```
时间参数化 (FAST/EGO):
    p(t) → 速度变化会改变路径形状
    低速时曲率计算不稳定

弧长参数化 (本工程):
    p(s), s(t) → 几何与速度解耦
    曲率 = f(s) 与速度无关
    停车时轨迹仍然定义良好
```

**2. 地形感知动力学 (Terrain-Aware Dynamics)**

```
常规 (FAST/EGO):
    v_max, a_max 是常数
    不考虑地形坡度

本工程:
    v_eff = v_max / inv_cos_φₓ     // 上坡速度降低
    a_eff = a_lon - g·sin_φₓ        // 坡度影响加速度
    a_lat_eff = a_lat - g·sin_φᵧ    // 横坡影响横向加速度
```

**3. Footprint 碰撞检测 (vs 单点检测)**

```
常规 (FAST/EGO):
    检查轨迹点中心的 ESDF 值
    无法处理车辆尺寸

本工程:
    对车辆 footprint 的 4 个角点分别检查
    p_corner = pos + R(θ) · corner_body
    d = SDF(p_corner)
    更准确地反映实际碰撞风险
```

**4. 多段换挡规划**

```
常规 (FAST/EGO):
    只支持前进
    无法处理需要倒车的场景

本工程:
    前端搜索支持前进/后退
    后端优化为每段分配独立的 MINCO 轨迹
    换挡点作为优化变量
    换挡代价 (14.0) 避免频繁切换
```

---

## 六、重规划状态机

### 6.1 状态转移

```
INIT → WAIT_TARGET → SEQUENTIAL_START → EXEC_TRAJ → REPLAN_TRAJ
  ↑                                                     │
  └─────────────────────────────────────────────────────┘
```

### 6.2 重规划触发条件

1. **轨迹过期**: 执行时间 > 轨迹时长 + 余量
2. **跟踪误差**: 预测位置与里程计偏差 > `track_err_replan_dist (0.55m)` 或航向偏差 > `track_err_replan_yaw (0.70 rad)`
3. **碰撞危险**: 前瞻 SDF 距离 < `safe_margin + collision_replan_margin`
4. **轨迹即将过期**: 预防性重规划

### 6.3 目标投影

当原始目标在障碍物内或超出地图范围时:

```
沿当前位置→目标方向搜索最佳可达子目标:
score = progress + 0.20·sdf - 0.35·risk - 0.30·lateral_dev
```

---

## 七、总结

本工程的 ARC-Planner 相比 FAST-Planner / EGO-Planner 的核心创新在于:

1. **面向 Ackermann 车辆**: 自行车运动学模型，非完整约束
2. **弧长参数化**: 几何与时间解耦，曲率计算稳定
3. **地形感知**: 坡度/曲率/台阶高度集成到优化约束中
4. **严格约束处理**: ALM 代替简单罚函数，约束满足更精确
5. **多段换挡**: 支持前进/后退混合规划
6. **Footprint 碰撞检测**: 比单点检测更准确
