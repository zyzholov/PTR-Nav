# 地形建图详细流程与原理

## 一、总体架构

地形分析器 (`Terrain Analyzer`) 是连接激光雷达感知与路径规划的核心模块。它将原始点云转化为三种互补的地图表示:

```
FAST_LIO2 输出
    │
    │ /cloud_registered (世界坐标系点云)
    │ /Odometry (位姿)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                  Terrain Analyzer                        │
│                                                         │
│  点云输入 ──┬──▶ 高程图 (Elevation Map)                  │
│             │        │                                  │
│             │        ├──▶ 风险图 (Risk Map)              │
│             │        │                                  │
│             └──▶ 障碍物检测 ──▶ 障碍物膨胀 ──▶ SDF Map   │
│                                                         │
│  输出:                                                  │
│  ├─ fused_map (高程 + 风险 + 法向量)                     │
│  └─ sdf_map (距离场)                                    │
└─────────────────────────────────────────────────────────┘
    │                    │
    ▼                    ▼
  规划器               MPC 控制器
```

**为什么需要三种地图？**

| 地图 | 回答的问题 | 消费者 |
|------|-----------|--------|
| 高程图 | "地面有多高？" | 风险计算、坡度估计 |
| 风险图 | "地面能走吗？" | 轨迹优化器 |
| SDF 图 | "离障碍物多远？" | A* 碰撞检测、轨迹避障 |

三者互补: **SDF 处理"硬障碍"（树干、墙壁），Risk 处理"软障碍"（陡坡、台阶），高程图是两者的共同基础。**

---

## 二、点云处理流水线

### 2.1 整体流程 (`cloudCallback`)

每次收到一帧点云时，执行以下处理:

```
输入: /cloud_registered (世界坐标系点云) + /Odometry (位姿)
    │
    ▼
步骤1: 滑动窗口重定位
    │   将地图中心移动到车辆当前位置
    │   保留已累积的历史数据
    │
    ▼
步骤2: 第一遍扫描 — 地表更新
    │   对每个点:
    │   ├─ Ego Body Filter (过滤车身内的点)
    │   ├─ 世界坐标 → 栅格索引
    │   └─ 更新高程图 (非对称 EMA)
    │
    ▼
步骤3: 第二遍扫描 — 障碍物检测
    │   对每个保留的点:
    │   ├─ 估计参考地面高度
    │   ├─ 判断是否在碰撞带内
    │   └─ 记录障碍物证据
    │
    ▼
步骤4: Footprint 过滤
    │   过滤密集的 2D 斑块 (可能是树冠裙边)
    │   保留线状结构 (树干、墙壁)
    │
    ▼
步骤5: 生命周期更新
    │   衰减未命中的障碍物计数
    │   清除过期的障碍物标记
    │
    ▼
步骤6: 发布触发
    └─ 调用 publishTerrain() 生成最终地图产品
```

### 2.2 滑动窗口机制

地图以车辆为中心，随车辆移动而滚动:

```
rollLiveAccumulatedMap():
    // 计算车辆位置与地图中心的偏移 (栅格单位)
    dx_cells = round((vehicle_x - map_center_x) / resolution)
    dy_cells = round((vehicle_y - map_center_y) / resolution)

    // 如果偏移非零，滚动所有图层
    if dx_cells != 0 or dy_cells != 0:
        for each layer in [elevation, top_elevation, variance,
                           obstacle_band_count, vertical_obstacle_map, ...]:
            shiftRollingLayer(layer, dx_cells, dy_cells)
        // 滚动后，新暴露的区域初始化为默认值
```

这保证了地图始终以车辆为中心，同时保留已观测区域的历史数据。

---

## 三、高程图构建

### 3.1 非对称指数移动平均 (Asymmetric EMA)

高程图使用 **非对称 EMA** 融合每个栅格的激光雷达观测。核心思想: **低点优先更新，高点几乎不动**。

对于每个落入栅格 `idx` 的点 `pt.z`:

**首次观测:**
```
elevation_map_[idx] = pt.z
top_elevation_map_[idx] = pt.z
variance_map_[idx] = min_var
point_count_map_[idx] = 1
```

**后续观测:**
```
ground_band = max(0.05, min(step_height_soft × 0.5, 0.15))

if pt.z < elevation_map_[idx]:
    // 新点低于当前地面估计 → 强烈拉低
    elevation = 0.85 × pt.z + 0.15 × elevation

else if pt.z ≤ elevation_map_[idx] + ground_band:
    // 新点在地面附近 → 微弱拉高
    elevation = 0.95 × elevation + 0.05 × pt.z

else:
    // 新点显著高于地面 → 不更新地面 (可能是障碍物)
    pass

// 最高点始终更新
top_elevation_map_[idx] = max(top_elevation, pt.z)

// 方差衰减
variance_map_[idx] = max(min_var, variance × 0.95)

// 计数器递增
point_count_map_[idx]++
```

### 3.2 为什么用非对称 EMA？

| 问题 | 对称 EMA 的缺陷 | 非对称 EMA 的优势 |
|------|----------------|------------------|
| 障碍物点污染地面 | 障碍物点会抬高地面估计 | 高点权重极低 (0.05)，几乎不影响 |
| 地面跟踪延迟 | 低点和高点权重相同 | 低点权重高 (0.85)，快速响应 |
| 树冠/植被穿透 | 阳光穿透树冠的点会干扰 | 远高于地面的点被完全忽略 |

**设计原理:**
- 激光雷达的最低返回通常代表真实地面
- 高返回可能是树干、树叶、墙壁等障碍物
- 通过非对称权重，高程图自动"穿透"植被，跟踪真实地面

### 3.3 地面带 (Ground Band)

```
ground_band = max(0.05, min(step_height_soft × 0.5, 0.15))
```

- `step_height_soft = 0.22m` → `ground_band = max(0.05, min(0.11, 0.15)) = 0.11m`
- 只有低于当前地面 + ground_band 的点才参与地面更新
- 远高于地面的点被完全忽略，避免树冠/屋檐点污染地面

### 3.4 地面法向量计算

在 `computeTraversabilityMetrics()` 中，使用中心差分法计算地面法向量:

```
// 四邻域高程
elev_l = elevation[row][col-1]
elev_r = elevation[row][col+1]
elev_d = elevation[row-1][col]
elev_u = elevation[row+1][col]

// 梯度 (中心差分)
grad_x = (elev_r - elev_l) / (2 × resolution)
grad_y = (elev_u - elev_d) / (2 × resolution)

// 法向量 (指向天空)
norm = sqrt(1 + grad_x² + grad_y²)
zbx = -grad_x / norm    // 法向量 x 分量
zby = -grad_y / norm    // 法向量 y 分量
zbz = 1 / norm          // 法向量 z 分量
```

法向量 `(zbx, zby, zbz)` 用于:
1. 坡度计算: `slope = sqrt(grad_x² + grad_y²)`
2. 地形感知动力学: 纵向/横向坡度因子修正速度和加速度

---

## 四、风险图构建

### 4.1 三个地形特征

风险图综合评估三个地形特征:

#### (a) 坡度 (Slope)

```
slope = sqrt(grad_x² + grad_y²)
```

物理意义: 地面的倾斜程度。坡度越大，车辆越容易打滑或翻覆。

#### (b) 曲率 (Curvature)

```
hess_xx = (elev_r - 2×elev + elev_l) / resolution²
hess_yy = (elev_u - 2×elev + elev_d) / resolution²
curvature = sqrt(hess_xx² + hess_yy²)
```

物理意义: 地面的凹凸程度。高曲率意味着地面有突然的凹陷或隆起，可能导致车辆底部托底。

#### (c) 台阶高度 (Step Height)

```
step_height = max(|elev - elev_l|, |elev - elev_r|,
                  |elev - elev_d|, |elev - elev_u|)
```

物理意义: 相邻栅格的最大高程差。台阶高度越大，车辆越难通过。

### 4.2 软阈值映射 (Soft Threshold + Gamma)

每个特征通过软阈值映射到 [0, 1] 分数:

```
softScore(value, free_value, soft_value) = clamp((value - free_value) / (soft_value - free_value), 0, 1)
```

然后施加 gamma 幂次:

```
score = pow(softScore(value, free, soft), gamma)
```

**阈值配置 (本工程实际值):**

| 特征 | `free` (score=0) | `soft` (score=1) | `hard` (硬障碍) |
|------|------------------|------------------|-----------------|
| 坡度 | `traversable_slope_free = 0.55 rad` (~31°) | `sdf_obstacle_slope = 1.10 rad` (~63°) | `sdf_hard_obstacle_slope = 1.65 rad` (~95°) |
| 曲率 | `traversable_curvature_free = 0.45` | `sdf_obstacle_curvature = 1.00` | `sdf_hard_obstacle_curvature = 1.80` |
| 台阶 | `traversable_step_free = 0.09m` | `step_height_soft = 0.22m` | `step_height_hard = 0.50m` |

**Gamma 的作用:**

```
gamma = traversability_risk_gamma = 2.40
```

- `gamma > 1`: 使分数曲线在阈值附近更陡峭，低风险区域保持低分，高风险区域快速升高
- `gamma < 1`: 使分数曲线更平缓，风险分布更均匀
- `gamma = 1`: 线性映射

```
score
  │
1 │            gamma=2.4 (本工程)
  │           ╱
  │          ╱
  │        ╱
  │      ╱   gamma=1.0 (线性)
  │    ╱   ╱
  │  ╱  ╱
  │╱ ╱
0 └──────────── value
   free    soft
```

### 4.3 加权合成风险值

```
risk = (w_slope × slope_score + w_curv × curv_score + w_step × step_score)
       / (w_slope + w_curv + w_step)
```

**风险权重 (本工程实际值):**
```
risk_weight_slope = 0.22
risk_weight_curvature = 0.33
risk_weight_step_height = 0.45
```

**权重分析:**
- 台阶高度权重最高 (0.45): 台阶对车辆影响最大，可能导致托底或卡住
- 曲率权重次之 (0.33): 凹凸地形影响悬挂和轮胎接触
- 坡度权重最低 (0.22): 车辆有一定爬坡能力

### 4.4 硬障碍物分类

当地形特征超过硬阈值时，该栅格被标记为硬障碍物:

```
hard_obstacle = (slope ≥ sdf_hard_obstacle_slope) ||
                (curvature ≥ sdf_hard_obstacle_curvature) ||
                (step_height ≥ step_height_hard)
```

硬障碍物栅格的 `risk` 被强制设为 `risk_hard_threshold (0.92)`。

### 4.5 树木风险场 (Tree Risk Field)

除了地形风险，还为每个垂直障碍物种子计算 **树木距离风险**:

```
// 使用 Chamfer 距离计算到最近树木种子的距离
tree_dist = chamfer_distance_to_nearest_vertical_seed(x, y)

// 风险随距离衰减
tree_risk = pow(clamp(1.0 - tree_dist / tree_risk_radius, 0, 1), tree_risk_power)
```

参数:
- `tree_risk_radius_m = 0.38m`: 风险衰减半径
- `tree_risk_power = 3.0`: 衰减指数 (3 次方衰减，近距离风险急剧升高)

**融合风险:**
```
fused_risk = max(terrain_risk_scale × terrain_risk, tree_risk)

// 非硬障碍物且非垂直种子的区域，风险有上限
if !hard_obstacle && !vertical_seed:
    fused_risk = min(fused_risk, non_tree_risk_cap)   // 0.05

// 硬障碍物区域，风险有下限
if hard_obstacle:
    fused_risk = max(fused_risk, risk_hard_threshold)  // 0.92
```

---

## 五、SDF 图构建

### 5.1 Chamfer 距离传播算法

SDF 使用 **3×3 Chamfer 距离** 的两遍扫描近似:

#### 初始化

```
for each cell (row, col):
    if obstacle_map[row][col] == 1:
        dist_field[row][col] = 0.0      // 障碍物表面
    else:
        dist_field[row][col] = +∞       // 自由空间
```

障碍物来源:
- `vertical_obstacle_map_`: 激光雷达检测到的垂直障碍物
- `terrain_hard_obstacle`: 地形硬障碍物 (可选，由 `sdf_use_terrain_hard_obstacles` 控制)

#### 前向传播 (左上 → 右下)

```
w_diag = √2 ≈ 1.414

for row = 0 to grid_size_y - 1:
    for col = 0 to grid_size_x - 1:
        idx = row × grid_size_x + col
        best = dist_field[idx]

        // 基数邻居 (权重 1.0)
        if col > 0:
            best = min(best, dist_field[idx - 1] + 1.0)
        if row > 0:
            best = min(best, dist_field[idx - grid_size_x] + 1.0)

        // 对角邻居 (权重 √2)
        if col > 0 and row > 0:
            best = min(best, dist_field[idx - grid_size_x - 1] + w_diag)
        if col + 1 < grid_size_x and row > 0:
            best = min(best, dist_field[idx - grid_size_x + 1] + w_diag)

        dist_field[idx] = best
```

#### 后向传播 (右下 → 左上)

```
for row = grid_size_y - 1 to 0:
    for col = grid_size_x - 1 to 0:
        idx = row × grid_size_x + col
        best = dist_field[idx]

        // 基数邻居 (权重 1.0)
        if col + 1 < grid_size_x:
            best = min(best, dist_field[idx + 1] + 1.0)
        if row + 1 < grid_size_y:
            best = min(best, dist_field[idx + grid_size_x] + 1.0)

        // 对角邻居 (权重 √2)
        if col + 1 < grid_size_x and row + 1 < grid_size_y:
            best = min(best, dist_field[idx + grid_size_x + 1] + w_diag)
        if col > 0 and row + 1 < grid_size_y:
            best = min(best, dist_field[idx + grid_size_x - 1] + w_diag)

        dist_field[idx] = best
```

#### 转换为米

```
sdf_distance = dist_field[idx] × resolution
sdf_distance = clamp(sdf_distance, 0.0, 20.0)   // 最大 20m
```

### 5.2 Chamfer 距离 vs 精确欧氏距离

| 方法 | 精度 | 计算复杂度 | 适用场景 |
|------|------|-----------|----------|
| 精确欧氏距离 (Felzenszwalb) | 精确 | O(N) | ROS1 路径 |
| Chamfer 3×3 | 近似 (误差 ≤ 8%) | O(N) | ROS2 路径 (本工程) |
| Chamfer 5×5 | 更近似 (误差 ≤ 2%) | O(N) | 可选 |

Chamfer 距离的优势:
- 实现极其简单 (两遍扫描)
- 内存访问模式友好 (缓存命中率高)
- 误差在规划场景中可接受

### 5.3 SDF 的作用

**在前端搜索 (Kino A*) 中:**
```
isCollided(state):
    d = SDF(state.x, state.y)
    return (d ≤ safe_dis)    // safe_dis = 0.42m
```

**在后端优化中:**
```
for each footprint corner:
    p_abs = pos + R(θ) · corner_body
    d = SDF(p_abs)
    violation = safe_margin - d
    if violation > 0:
        cost += esdfWei × smoothedL1(violation)
```

SDF 提供了连续的距离信息，使得:
1. 碰撞检测: `d ≤ 0` 表示在障碍物内
2. 安全裕度: `d ≤ safe_margin` 表示太近
3. 避障梯度: SDF 的梯度指向远离障碍物的方向

---

## 六、垂直障碍物检测

### 6.1 碰撞带判断 (`isCollisionBandPoint`)

判断一个点是否为障碍物种子:

```
// 估计参考地面 (邻域 25 分位数)
reference_ground = estimateObstacleReferenceGround(point)

// 相对高度
rel_h = point.z - reference_ground

// 高度带测试
min_h = obstacle_min_height = 0.28m
max_h = obstacle_max_height = 1.45m
seed_max_h = obstacle_seed_max_height = 0.50m

if rel_h ≤ min_h or rel_h ≥ max_h:
    REJECT  // 太低 (地面) 或太高 (噪声/头顶)

if rel_h > seed_max_h:
    REJECT  // 太高，不适合作为种子 (可能是树冠)
```

### 6.2 垂直结构测试 (反树冠过滤)

这是防止树冠裙边被误判为障碍物的关键过滤:

```
// 同一栅格内的高度跨度
same_cell_span = point.z - cell_ground

// 栅格地面与参考地面的差距
base_gap = cell_ground - reference_ground

// 垂直结构门槛
vertical_gate = max(0.16, min_h × 0.80) = max(0.16, 0.224) = 0.224m

// 必须满足以下条件之一:
has_vertical_structure =
    (same_cell_span > vertical_gate)           // 同一栅格内有显著高度跨度
    OR (neighbor_support > 0 AND base_gap > vertical_gate)  // 邻域有支撑且基座抬高
```

**设计原理:**
- 树干: 同一栅格内从地面到树干顶部有显著高度跨度 → `same_cell_span > gate`
- 墙壁: 邻域有支撑 (相邻栅格也有障碍物) 且基座抬高 → `neighbor_support > 0 AND base_gap > gate`
- 树冠裙边: 散布在较大区域内，单个栅格内跨度小，基座不抬高 → 被过滤

### 6.3 参考地面估计 (`estimateObstacleReferenceGround`)

```
// 搜索邻域 (半径 obstacle_ground_search_radius_m = 0.32m)
neighbor_grounds = []
for each neighbor cell within radius:
    if point_count_map[neighbor] >= min_points_for_traversability:
        neighbor_grounds.append(elevation_map[neighbor])

// 取 25 分位数 (或第一个元素)
if neighbor_grounds.size() >= 5:
    sort(neighbor_grounds)
    reference_ground = neighbor_grounds[25th percentile index]
else:
    reference_ground = neighbor_grounds[0]
```

使用 25 分位数而非最小值的原因:
- 最小值可能是噪声或异常低点
- 25 分位数更稳健，同时仍偏向低地面

### 6.4 Footprint 过滤 (`acceptObstacleSeedFootprint`)

过滤密集的 2D 斑块 (可能是树冠裙边或树叶返回):

```
// 搜索邻域
hit_count = 0
for each cell within obstacle_footprint_filter_radius_m (0.45m):
    if raw_obstacle_hit_map[neighbor] > 0:
        hit_count++

// 如果邻居太多，检查是否为线状结构
if hit_count > obstacle_footprint_max_neighbors (10):
    // 计算斑块的空间范围
    x_span = max_x - min_x
    y_span = max_y - min_y

    // 线状结构: 在一个方向上很窄
    is_line_like = (x_span ≤ line_width_cells) OR (y_span ≤ line_width_cells)

    if !is_line_like:
        REJECT  // 密集斑块，可能是树冠
```

**保留的结构:**
- 树干: 紧凑的点，邻居数 ≤ 10
- 墙壁: 线状结构，一个方向跨度 ≤ 2 格

**过滤的结构:**
- 树冠裙边: 大面积散布，邻居数 > 10，非线状

### 6.5 障碍物生命周期管理

#### 记录障碍物证据 (`recordCollisionEvidence`)

```
// 每次命中
cnt = min(65535, obstacle_band_count_map_[idx] + obstacle_hit_increment)
obstacle_band_count_map_[idx] = cnt

// 更新碰撞顶部高度
collision_top_elevation_map_[idx] = max(collision_top_elevation_map_[idx], point.z)

// 如果累积足够，立即标记为垂直障碍物
if cnt ≥ obstacle_min_points:
    vertical_obstacle_map_[idx] = 1
```

#### 生命周期更新 (`updateObstacleEvidenceLifecycle`)

每帧点云处理后，更新所有栅格的障碍物状态:

```
for each cell i:
    if 本帧有命中:
        obstacle_miss_count_map_[i] = 0  // 重置未命中计数
    else:
        obstacle_miss_count_map_[i]++    // 递增未命中计数

        // 衰减障碍物计数
        cnt = floor(obstacle_band_count_map_[i] × obstacle_count_decay)
        obstacle_band_count_map_[i] = max(0, cnt - 1)  // 确保单调递减

    // 判断是否仍然活跃
    has_collision_top = isfinite(collision_top_elevation_map_[i])

    active_by_confidence =
        has_collision_top AND
        obstacle_band_count_map_[i] ≥ obstacle_min_points

    active_by_persistence =
        was_active AND
        has_collision_top AND
        miss_count ≤ obstacle_persistence_misses

    active = active_by_confidence OR active_by_persistence
    vertical_obstacle_map_[i] = active ? 1 : 0

    // 完全过期时清除
    if cnt == 0 AND miss_count > obstacle_persistence_misses:
        collision_top_elevation_map_[i] = -∞
```

**生命周期参数 (本工程实际值):**
| 参数 | 值 | 说明 |
|------|-----|------|
| `obstacle_hit_increment` | 4 | 每次命中增加的计数 |
| `obstacle_count_decay` | 0.98 | 每帧衰减系数 |
| `obstacle_min_points` | 2 | 最小计数才标记为障碍物 |
| `obstacle_persistence_misses` | 18 | 连续未命中帧数后清除 |

**设计原理:**
- 衰减机制: 障碍物如果不再被观测到，会逐渐"淡出"
- 持久性机制: 已确认的障碍物在短期内未命中时仍保持活跃
- 双重判定: 置信度路径 (高计数) 和持久性路径 (低未命中) 互补

---

## 七、Ego Body Filter (本体过滤器)

### 7.1 过滤逻辑

在车身坐标系中定义一个 3D 盒子，过滤掉扫描到车辆自身的点:

```
// 转换到车身坐标系
dx = pt.x - vehicle_x
dy = pt.y - vehicle_y
bx = cos(yaw)×dx + sin(yaw)×dy     // 纵向
by = -sin(yaw)×dx + cos(yaw)×dy    // 横向
bz = pt.z - vehicle_z               // 垂直

// 判断是否在车身内
if bx ≤ ego_body_filter_front_m AND
   bx ≥ -ego_body_filter_back_m AND
   |by| ≤ ego_body_filter_half_width_m AND
   bz ≥ ego_body_filter_z_min_m AND
   bz ≤ ego_body_filter_z_max_m:
    FILTER OUT  // 丢弃该点
```

### 7.2 参数配置 (本工程实际值)

```yaml
ego_body_filter_enable: true
ego_body_filter_front_m: 0.50      # 前方
ego_body_filter_back_m: 0.50       # 后方
ego_body_filter_half_width_m: 0.30 # 半宽
ego_body_filter_z_min_m: -0.55     # 垂直下限
ego_body_filter_z_max_m: 1.40      # 垂直上限
```

过滤区域: 前后各 0.5m，左右各 0.3m 的矩形。

### 7.3 为什么需要 Ego Body Filter？

- 激光雷达安装在车身上，会扫描到车身结构 (轮毂、保险杠、车顶等)
- 这些点的高程与地面不同，会污染高程图
- 如果不过滤，车身周围的栅格会被误判为有障碍物

### 7.4 盲区问题

Ego Body Filter 同时也过滤掉了车身附近的 **真实障碍物** 点。这创造了一个检测盲区:

```
        ┌───────────────────┐
        │   Ego Body Filter │
        │   (检测盲区)       │
        │                   │
        │    ┌─────────┐    │
        │    │  车辆    │    │
        │    └─────────┘    │
        │                   │
        └───────────────────┘
              ↑
              这里的障碍物无法被检测到
```

**缓解措施:**
- 缩小 Ego Body Filter 尺寸 (本工程已调整为 0.50m)
- 使用 Ego Clear 保留近距离的真实障碍物证据

---

## 八、障碍物膨胀与 Ego Clear

### 8.1 障碍物膨胀

为每个垂直障碍物种子创建安全裕度:

```
inflate_cells = ceil(vertical_obstacle_inflation_m / resolution)
             = ceil(0.41 / 0.20) = 3 格

for each cell where vertical_seed_map[idx] == 1:
    for dy = -inflate_cells to +inflate_cells:
        for dx = -inflate_cells to +inflate_cells:
            if dx² + dy² ≤ inflate_cells²:
                inflated_obstacle_map[row+dy][col+dx] = 1
```

**膨胀的作用:**
- 为车辆提供安全裕度
- 膨胀半径 0.41m ≈ 车辆半宽 0.30m + 安全余量 0.11m
- 确保车辆可通过的间隙不会被完全堵死

### 8.2 Ego Clear (车辆本体清除)

清除车辆本体范围内的障碍物标记，防止自检测干扰:

```
// 清除区域 (比 Ego Body Filter 更紧凑的矩形)
clear_front = clamp(0.32 × ego_body_filter_front, 0.18, ego_clear_radius)
clear_back  = clamp(0.35 × ego_body_filter_back, 0.16, ego_clear_radius)
clear_half_width = clamp(0.42 × ego_body_filter_half_width, 0.12, ego_clear_radius × 0.65)

// 保留逻辑: 近距离的真实障碍物证据不被清除
preserve_radius = 0.12m

for each cell in clear region:
    has_obstacle_evidence =
        vertical_seed_map[idx] OR
        obstacle_map[idx] OR
        obstacle_band_count_map[idx] ≥ min_points OR
        isfinite(collision_top_elevation_map[idx])

    // 如果有证据且在保留半径外，保留
    if has_obstacle_evidence AND distance > preserve_radius:
        CONTINUE  // 不清除

    // 否则清除所有障碍物相关数据
    obstacle_map[idx] = 0
    vertical_seed_map[idx] = 0
    vertical_obstacle_map[idx] = 0
    obstacle_band_count_map[idx] = 0
    collision_top_elevation_map[idx] = -∞
```

**设计原理:**
- 宽的圆形清除区域会在车辆接近障碍物时擦除障碍物证据
- 导致 SDF/Risk 看起来可通行，规划器会撞上障碍物
- 紧凑的矩形 + 0.12m 保留环缓解了这个问题

---

## 九、发布流水线 (`publishTerrain`)

### 9.1 核心地形计算

对每个栅格:

```
1. 采样高程和方差
   elev = elevation_map_[idx]
   var  = variance_map_[idx]

2. 计算可通行性指标
   computeTraversabilityMetrics(elev, neighbors)
   → slope, curvature, step_height, risk, zbx, zby, hard_obstacle

3. 合并障碍物来源
   is_obstacle = vertical_seed OR (hard_obstacle AND sdf_use_terrain_hard_obstacles)

4. 垂直种子的 risk 强制为 1.0
   if vertical_seed: risk = 1.0

5. 写入 SE2Grid
   fused_map["elevation"][idx] = elev
   fused_map["risk"][idx] = risk        // 对所有 yaw 索引
   fused_map["zbx"][idx] = zbx
   fused_map["zby"][idx] = zby
```

### 9.2 SDF 构建

```
1. 初始化: 障碍物栅格 = 0, 自由栅格 = +∞
2. 前向传播 (左上→右下)
3. 后向传播 (右下→左上)
4. 转换为米: sdf = dist × resolution
5. 写入 SE2Grid: sdf_map["sdf_cpu"][idx] = sdf
```

### 9.3 发布的话题

| 话题 | 数据格式 | 内容 | 频率 |
|------|----------|------|------|
| `fused_map` | SE2Grid | 高程 + 风险 + 法向量 + 方差 | 15Hz |
| `sdf_map` | SE2Grid | SDF 距离场 | 15Hz |
| `elevation_cloud` | PointCloud2 | 高程可视化 (Jet 色图) | 9Hz |
| `sdf_cloud` | PointCloud2 | SDF 可视化 (绿=远, 红=近) | 9Hz |
| `risk_cloud` | PointCloud2 | 风险可视化 (蓝=低, 红=高) | 9Hz |
| `obstacle_cloud` | PointCloud2 | 障碍物可视化 (红=垂直种子, 橙=硬障碍) | 9Hz |
| `overhead_cloud` | PointCloud2 | 头顶物体可视化 (绿色) | 9Hz |

---

## 十、SE2Grid 数据结构

### 10.1 网格组织

SE2Grid 是一个多层 2.5D 栅格库，支持可选的 yaw 离散化:

```
geometry:
    length_x = grid_size_x × resolution = 104 × 0.2 = 20.8m
    length_y = grid_size_y × resolution = 104 × 0.2 = 20.8m
    resolution = 0.2m
    yaw_resolution = 6.0° (可选)

存储: 循环缓冲区 (circular buffer)
    start_index: 随车辆移动而滚动
    数据布局: 行优先 (row-major)
    索引转换: toSe2Col(x) = grid_size_x - 1 - x
```

### 10.2 图层定义

**Fused Map 图层:**

| 图层 | Yaw 离散化 | 数据类型 | 说明 |
|------|-----------|----------|------|
| `elevation` | 否 | float | 地面高程 (m) |
| `var` | 否 | float | 高程方差 |
| `risk` | 是 | float | 可通行性风险 [0, 1] |
| `valid` | 是 | bool | 有效性标志 |
| `zbx` | 是 | float | 法向量 x 分量 |
| `zby` | 是 | float | 法向量 y 分量 |

**SDF Map 图层:**

| 图层 | Yaw 离散化 | 数据类型 | 说明 |
|------|-----------|----------|------|
| `sdf_cpu` | 否 | float | Chamfer 距离 (m) |
| `valid` | 否 | bool | 有效性标志 |

### 10.3 为什么 Risk 和法向量需要 Yaw 离散化？

- `elevation` 和 `sdf` 与车辆朝向无关 → 不需要 yaw 离散化
- `risk` 和 `zbx/zby` 在优化器中按 yaw 索引查询 → 需要 yaw 离散化以支持插值

---

## 十一、与常见地形建图方案的对比

| 特性 | 本工程 (Terrain Analyzer) | 传统高程图 | 体素地图 |
|------|--------------------------|-----------|---------|
| 数据结构 | 2.5D 栅格 (SE2Grid) | 2D 栅格 | 3D 体素 |
| 地面估计 | 非对称 EMA (低点优先) | 最小值/平均值 | 多层高度 |
| 障碍物表示 | 碰撞带 + 生命周期 | 二值标记 | 占据概率 |
| 距离场 | Chamfer 2-pass | 无/后处理 | 3D ESDF |
| 风险评估 | 坡度+曲率+台阶 加权 | 无 | 无 |
| 实时性 | 15Hz (CPU) | 高 | 低 (3D 计算量大) |
| 内存 | ~104×104×6 层 ≈ 2.5MB | ~104×104×3 层 ≈ 1.3MB | 104³ 体素 ≈ 1GB |

**本工程的设计优势:**
1. 非对称 EMA 自动"穿透"植被，跟踪真实地面
2. 碰撞带检测 + 生命周期管理，稳健地区分地面和障碍物
3. 三种地图互补，覆盖"硬障碍"和"软障碍"
4. Chamfer SDF 计算高效，适合实时规划

**设计权衡:**
1. 2.5D 表示无法处理多层结构 (如桥下道路)
2. Chamfer SDF 精度不如精确欧氏距离 (~8% 误差)
3. 非对称 EMA 在快速变化地形中可能有延迟
