# ARM 架构平台移植操作指南

## 一、移植概述

### 1.1 为什么要移植到 ARM？

当前工程在 x86_64 PC（Gazebo 仿真）上开发完成，需要移植到 ARM 架构的车载计算平台进行实车部署。常见的 ARM 车载平台包括：

| 平台 | 架构 | GPU | 典型用途 |
|------|------|-----|----------|
| NVIDIA Jetson Orin NX | aarch64 | NVIDIA (CUDA) | 机器人/自动驾驶边缘计算 |
| NVIDIA Jetson AGX Xavier | aarch64 | NVIDIA (CUDA) | 高性能边缘计算 |
| 华为 Atlas 200 | aarch64 | 昇腾 NPU | 国产化方案 |
| RK3588 开发板 | aarch64 | Mali GPU | 低成本方案 |
| 工控机 (ARM 服务器) | aarch64 | 可选 | 工业部署 |

### 1.2 移植工作量评估

```
移植工作分为四个层面：

1. 系统环境搭建         → 1-2 天
2. 依赖库安装与编译     → 1-2 天
3. 源码修改             → 0.5 天（少量修改）
4. 编译调试与性能优化   → 2-3 天
```

### 1.3 关键发现总结

经过对工程全部源码的架构分析，发现以下关键问题：

| 优先级 | 问题 | 影响 | 解决方案 |
|--------|------|------|----------|
| 高 | `ekf_quat_pose` 使用 `-march=native` | 交叉编译失败 | 删除或改为架构判断 |
| 高 | Livox SDK 静态库为 x86 编译 | 链接失败 | 在 ARM 上重新编译 Livox SDK |
| 高 | `FAST_LIO` CMake 中 x86 架构判断 | ARM 上禁用多线程 | 修改架构判断逻辑 |
| 中 | CasADi 需要 ARM 版本 | MPC 无法编译 | pip 安装或源码编译 |
| 中 | Gazebo include 路径硬编码 | 可能找不到头文件 | 修改为动态查找 |
| 低 | RapidJSON SIMD 代码 | 无影响 | 已内置 ARM NEON 回退 |
| 低 | CUDA 代码（可选） | 取决于平台 | 当前未启用，CPU 路径可用 |

---

## 二、系统环境搭建

### 2.1 操作系统

推荐使用 Ubuntu 22.04 aarch64，与 ROS2 Humble 官方支持的平台一致。

```bash
# 检查架构
uname -m
# 应输出: aarch64

# 检查系统版本
cat /etc/os-release
# 应包含: Ubuntu 22.04
```

**Jetson 用户注意：** JetPack 6.x 基于 Ubuntu 22.04，自带 CUDA 和 NVIDIA 驱动，无需单独安装。

### 2.2 ROS2 Humble 安装

```bash
# 设置 locale
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# 添加 ROS2 源
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
  sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 安装 ROS2 Humble（桌面版，含 RViz2 和 Gazebo）
sudo apt update && sudo apt upgrade
sudo apt install ros-humble-desktop

# 安装开发工具
sudo apt install python3-colcon-common-extensions python3-rosdep python3-vcstool

# 配置环境
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 2.3 核心依赖库安装

```bash
# Eigen3（纯头文件库，架构无关）
sudo apt install libeigen3-dev

# PCL（点云库）
sudo apt install libpcl-dev

# OpenCV 4
sudo apt install libopencv-dev

# Boost
sudo apt install libboost-all-dev

# Protobuf（Gazebo 依赖）
sudo apt install libprotobuf-dev protobuf-compiler

# OpenMP（GCC 默认支持，确认即可）
sudo apt install libomp-dev

# Qt5（RViz2 依赖）
sudo apt install qtbase5-dev

# tf2 相关
sudo apt install ros-humble-tf2 ros-humble-tf2-ros ros-humble-tf2-geometry-msgs

# PCL ROS2 桥接
sudo apt install ros-humble-pcl-ros ros-humble-pcl-conversions

# cv_bridge
sudo apt install ros-humble-cv-bridge

# Gazebo Classic（仿真用，实车可选）
sudo apt install ros-humble-gazebo-ros-pkgs

# 其他 ROS2 依赖
sudo apt install \
  ros-humble-robot-state-publisher \
  ros-humble-xacro \
  ros-humble-ackermann-msgs \
  ros-humble-message-filters \
  ros-humble-rosbag2 \
  ros-humble-pluginlib
```

### 2.4 CasADi 安装（MPC 求解器）

CasADi 是 MPC 控制器的核心依赖，必须安装 ARM 版本。

**方法一：pip 安装（推荐，最简单）**

```bash
pip3 install casadi
# 安装后设置环境变量
export CASADI_PREFIX=$(python3 -c "import casadi; import os; print(os.path.dirname(os.path.dirname(casadi.__file__)))")
echo "export CASADI_PREFIX=$CASADI_PREFIX" >> ~/.bashrc
```

**方法二：源码编译（更可控）**

```bash
# 安装编译依赖
sudo apt install cmake gfortran swig liblapack-dev

# 克隆 CasADi
cd ~ && git clone https://github.com/casadi/casadi.git
cd casadi && git checkout 3.6.5  # 使用稳定版本

# 编译
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DWITH_PYTHON=ON \
  -DWITH_PYTHON3=ON \
  -DWITH_IPOPT=ON \
  -DWITH_THREAD=ON
make -j$(nproc)
sudo make install

# 设置环境变量
export CASADI_PREFIX=/usr/local
echo "export CASADI_PREFIX=/usr/local" >> ~/.bashrc
```

**验证安装：**

```bash
python3 -c "import casadi; print(casadi.__version__)"
# 应输出版本号，如 3.6.5

# 验证 C++ 库
ls /usr/local/lib/libcasadi*
# 应有 libcasadi.so 等文件

ls /usr/local/include/casadi/
# 应有 casadi.hpp 等头文件
```

---

## 三、源码修改清单

### 3.1 必须修改的文件

#### 修改 1：`ekf_quat_pose/CMakeLists.txt` — 删除 `-march=native`

**文件：** `src/tank_slam/ekf_quat_pose/CMakeLists.txt`，第 7 行

**修改前：**
```cmake
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -DEIGEN_DONT_PARALLELIZE")
```

**修改后：**
```cmake
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_DONT_PARALLELIZE")
```

**原因：** `-march=native` 在 x86 机器上编译会生成 SSE/AVX 指令，交叉编译到 ARM 时会失败。即使是原生 ARM 编译，该标志也可能导致在不同 ARM 芯片之间不兼容。删除后 GCC 会使用默认的通用指令集。

---

#### 修改 2：`FAST_LIO/CMakeLists.txt` — ARM 多线程支持

**文件：** `src/FAST_LIO/CMakeLists.txt`，第 21-41 行

**修改前：**
```cmake
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  include(ProcessorCount)
  ProcessorCount(N)
  ...
  if(N GREATER 4)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=3)
  ...
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()
```

**修改后：**
```cmake
include(ProcessorCount)
ProcessorCount(N)
message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}, cores: ${N}")

if(N GREATER 4)
  add_definitions(-DMP_EN)
  add_definitions(-DMP_PROC_NUM=3)
elseif(N GREATER 3)
  add_definitions(-DMP_EN)
  add_definitions(-DMP_PROC_NUM=2)
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()
```

**原因：** 原始代码只在 x86 上启用 OpenMP 多线程，ARM 平台直接回退到单线程。Jetson Orin NX 有 8 核 CPU，完全支持 OpenMP。修改后 ARM 也能利用多核加速点云匹配。

---

#### 修改 3：`livox_laser_simulation_ros2/CMakeLists.txt` — Gazebo 路径

**文件：** `src/simulator/livox_laser_simulation_ros2/CMakeLists.txt`，第 39 行

**修改前：**
```cmake
include_directories(/usr/include/gazebo-11/gazebo)
```

**修改后：**
```cmake
find_package(gazebo REQUIRED)
include_directories(${GAZEBO_INCLUDE_DIRS})
```

**原因：** 硬编码路径假设 Gazebo 11 安装在 `/usr/include/gazebo-11/`，ARM 平台上路径可能不同。使用 CMake 的 `find_package` 可以自动找到正确路径。

**注意：** 如果实车不需要 Gazebo 仿真，可以跳过此修改，在编译时排除仿真相关包。

---

### 2.2 需要重新编译的外部库

#### Livox SDK（激光雷达驱动依赖）

Livox ROS Driver 依赖预编译的静态库 `liblivox_lidar_sdk_static.a`，x86 版本无法在 ARM 上链接。

```bash
# 方法一：让 CMake 自动下载编译（推荐）
# livox_ros_driver2 的 CMakeLists.txt 中已有自动 clone 逻辑
# 只要删除旧的 x86 静态库即可触发重新编译
sudo rm -f /usr/local/lib/liblivox_lidar_sdk_static.a
sudo rm -f /usr/local/lib/liblivox_sdk_static.a

# 方法二：手动编译 Livox SDK2
cd ~ && git clone https://github.com/Livox-SDK/Livox-SDK2.git
cd Livox-SDK2 && mkdir build && cd build
cmake .. && make -j$(nproc)
sudo make install

# 方法三：手动编译 Livox SDK1（livox_ros_driver 需要）
cd ~ && git clone https://github.com/Livox-SDK/Livox-SDK.git
cd Livox-SDK && mkdir build && cd build
cmake .. && make -j$(nproc)
sudo make install
```

---

### 2.3 不需要修改的文件（已验证可移植）

| 组件 | 说明 |
|------|------|
| RapidJSON (livox 驱动内) | 已内置 ARM NEON 指令回退，自动检测架构 |
| ikd-Tree (FAST_LIO) | 纯 C++ + pthread，无 SIMD，完全可移植 |
| Eigen3 | 纯头文件库，架构无关 |
| PCL | Ubuntu 官方 aarch64 包，直接 apt 安装 |
| OpenCV 4 | Ubuntu 官方 aarch64 包，直接 apt 安装 |
| 所有 YAML 配置文件 | 无平台相关路径，参数可直接复用 |
| Launch 文件 | `/opt/ros/humble/lib` 路径在 ARM 上相同 |
| MPC 控制器核心代码 | CasADi 接口架构无关，只需 ARM 版 CasADi |
| 地形分析器 CPU 路径 | `terrain_analyzer_ros2_node.cpp` 纯 CPU 实现 |

---

## 四、编译操作

### 4.1 清理旧的 x86 编译产物

**重要：** 不能将 x86 的 `build/`、`install/` 目录复制到 ARM 平台，必须从头编译。

```bash
cd /home/z/workspace_zyz/seb_clean  # 或 ARM 上对应的工作空间路径

# 删除旧的编译产物
rm -rf build/ install/ log/

# 如果有 Python venv，也需要重建
rm -rf src/.venv
python3 -m venv src/.venv
source src/.venv/bin/activate
pip3 install python-pptx  # 按需安装
```

### 4.2 全量编译

```bash
cd /home/z/workspace_zyz/seb_clean

# 确保没有 Anaconda 干扰
unset PYTHONHOME PYTHONPATH

# source ROS2
source /opt/ros/humble/setup.bash

# 全量编译
colcon build --symlink-install 2>&1 | tee build.log

# 如果编译成功
source install/setup.bash
```

### 4.3 局部编译（调试用）

```bash
# 只编译核心模块（不含仿真）
colcon build --packages-select \
  fast_lio \
  terrain_analyzer \
  path_search \
  plan_manage \
  mpc_controller \
  tank_sdk \
  se2_grid_core \
  se2_grid_ros \
  --symlink-install

source install/setup.bash
```

### 4.4 常见编译错误与解决

| 错误 | 原因 | 解决 |
|------|------|------|
| `undefined reference to 'casadi::...'` | CasADi 未安装或路径错误 | 检查 `CASADI_PREFIX` 环境变量 |
| `cannot find -llivox_lidar_sdk_static` | Livox SDK 未编译 ARM 版 | 重新编译 Livox SDK |
| `illegal instruction` 运行时崩溃 | `-march=native` 生成了不兼容指令 | 检查是否删除了所有 `-march=native` |
| `fatal error: gazebo/gazebo.hh: No such file` | Gazebo 路径硬编码 | 修改 CMake 使用 `find_package` |
| `catkin: command not found` | livox_ros_driver 依赖 ROS1 | 实车用 FAST_LIO 内置驱动即可 |
| `Python.h: No such file` | Python 开发头文件缺失 | `sudo apt install python3-dev` |

---

## 五、CUDA / GPU 相关处理

### 5.1 当前状态

地形分析器源码中包含 CUDA 版本（`TerrainBank.cu`），但 **当前 CMakeLists.txt 只编译 CPU 版本**（`terrain_analyzer_ros2_node.cpp`）。因此：

- **如果不需要 GPU 加速地形分析：** 无需任何 CUDA 配置，CPU 版本可直接编译运行。
- **如果需要 GPU 加速：** 需要额外配置 CUDA 环境。

### 5.2 Jetson 平台启用 CUDA（可选）

```bash
# JetPack 已自带 CUDA，确认版本
nvcc --version
# JetPack 6.x 对应 CUDA 12.x

# 如果要编译 CUDA 版本的地形分析器
# 需要修改 terrain_analyzer/CMakeLists.txt，添加：
#   enable_language(CUDA)
#   find_package(CUDA REQUIRED)
#   add_executable(terrain_analyzer_gpu_node
#     src/terrain_analyzer_gpu_node.cpp
#     src/TerrainBank.cu
#   )

# 注意：TerrainBank.cu 中使用了 Eigen 类型作为 CUDA kernel 参数
# 需要确保 Eigen 版本支持 CUDA（Eigen 3.4+ 已支持）
```

### 5.3 非 NVIDIA ARM 平台

如果目标平台没有 NVIDIA GPU（如 RK3588、华为 Atlas 等）：

- CUDA 代码无法使用，但 CPU 版本完全可用
- 地形分析器的 CPU 实现已经足够实时（15Hz）
- 如果需要 GPU 加速，需要将 CUDA kernel 移植为 OpenCL 或平台特定的 GPU 计算框架

---

## 六、平台特定配置

### 6.1 Jetson Orin NX 配置

```bash
# 1. 设置性能模式
sudo nvpmodel -m 0        # MAXN 模式
sudo jetson_clocks         # 锁定最高频率

# 2. 增加 swap（编译大项目时内存可能不够）
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 3. 确认 CPU 核心数
nproc
# Orin NX 应输出 8

# 4. 确认内存
free -h
# Orin NX 16GB 版应有约 15GB 可用
```

### 6.2 串口权限（连接底盘用）

```bash
# NEOS 底盘通过串口通信
# 确认串口设备
ls /dev/ttyUSB* /dev/ttyACM*

# 添加串口权限
sudo usermod -aG dialout $USER
# 需要重新登录生效

# 或临时设置
sudo chmod 666 /dev/ttyUSB0
```

### 6.3 网络配置（多机通信）

如果 PC 运行 RViz，ARM 板运行节点：

```bash
# ARM 板
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

# PC（同一局域网）
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

# 确认通信
# ARM 板上
ros2 topic list
# PC 上应该能看到相同的 topic 列表
```

---

## 七、实车部署参数适配

从仿真迁移到实车，需要修改以下配置参数：

### 7.1 MPC 参数

**文件：** `src/mpc_controller/params/controller_hunter_gazebo.yaml`

```yaml
# 轴距（必须实测！仿真用 0.55m，NEOS 图纸标注 0.430m）
mpc/wheel_base: 0.430

# 速度限制（实车要保守）
mpc/max_speed: 0.35        # 仿真 0.95，实车降为 0.35
mpc/min_speed: -0.10       # 倒车速度限制
mpc/reverse_speed_cap: 0.10

# 转向限制（NEOS 转向角更小）
mpc/max_steer: 0.31        # 仿真 0.5236 (30°)，NEOS 约 17.8°

# 控制周期（根据实际计算能力调整）
mpc/dt: 0.10               # 100ms，如果 ARM 算力不足可改为 0.15
mpc/predict_steps: 10      # 预测步数

# cmd_vel 语义（NEOS 底盘用转角，不是 yaw rate）
mpc/cmd_is_yaw_rate: false

# cmd_vel topic（NEOS 驱动订阅 /cmd_vel，不是 /hunter_se/cmd_vel）
# 在 launch 文件中通过 cmd_topic 参数设置
```

### 7.2 地形分析器参数

**文件：** `src/uneven_mapping/terrain_analyzer/params/hunter_fastlio_registered_live.yaml`

```yaml
# 障碍物高度阈值（NEOS 底盘更低，离地间隙 0.135m）
obstacle_min_height: 0.12    # 仿真 0.28，实车降低

# 台阶阈值
step_height_soft: 0.06       # 仿真 0.22，实车降低
step_height_hard: 0.15       # 仿真 0.50，实车降低

# 本体过滤器（根据 NEOS 实际尺寸调整）
ego_body_filter_front_m: 0.40
ego_body_filter_back_m: 0.40
ego_body_filter_half_width_m: 0.28
ego_clear_radius: 0.25
```

### 7.3 规划器参数

**文件：** `src/planner/arc_planner/plan_manage/config/planning_hunter_gazebo.yaml`

```yaml
# 最大速度（实车保守）
vmax: 0.35                   # 仿真 0.70，实车减半

# 最大曲率（根据 NEOS 最小转弯半径 1.34m 计算）
kmax: 0.75                   # 1/1.34 ≈ 0.75

# 安全裕度（实车增大）
safeMargin: 0.50             # 仿真 0.42，实车增大
```

### 7.4 FAST_LIO 参数

**文件：** `src/FAST_LIO/config/gazebo_hunter_mid360.yaml`

```yaml
# 如果实车 FAST_LIO 建图质量不好，可以调整：
# - 检查 LiDAR-IMU 外参是否标定正确
# - 调整 filter_size_surf（点云滤波尺寸）
# - 谁整 outlier_threshold（离群点阈值）
```

---

## 八、启动脚本适配

### 8.1 创建 ARM 平台启动脚本

```bash
#!/bin/bash
# run_neos_real.sh — NEOS 实车启动脚本

cd /home/z/workspace_zyz/seb_clean  # ARM 上的工作空间路径

unset PYTHONHOME PYTHONPATH
source /opt/ros/humble/setup.bash
source install/setup.bash

# 启动 FAST_LIO（连接真实 Livox MID360）
ros2 launch fast_lio mapping.launch.py \
  config_file:=neos_real.yaml \
  use_sim_time:=false &

# 等待 FAST_LIO 初始化
sleep 5

# 启动地形分析 + 规划 + MPC（不开 Gazebo 和 RViz）
ros2 launch plan_manage hunter_gazebo_fastlio_nav.launch.py \
  terrain:=true \
  planner:=true \
  mpc:=true \
  rviz:=false \
  cmd_topic:=/cmd_vel \
  odom_topic:=/Odometry \
  use_sim_time:=false
```

### 8.2 Launch 文件修改

**文件：** `src/planner/arc_planner/plan_manage/launch/hunter_gazebo_fastlio_nav.launch.py`

需要支持 `cmd_topic` 参数传递给 MPC 节点：

```python
# 在 launch 文件中添加参数传递
cmd_topic = LaunchConfiguration('cmd_topic', default='/cmd_vel')

# MPC 节点的 remapping
Node(
    package='mpc_controller',
    executable='mpc_controller_node',
    remappings=[
        ('/hunter_se/cmd_vel', cmd_topic),
    ],
    parameters=[...],
)
```

---

## 九、编译验证检查清单

在 ARM 平台上完成编译后，按以下清单逐项验证：

### 9.1 编译检查

```bash
# 1. 全量编译无错误
colcon build --symlink-install 2>&1 | grep -E "error|Error|FAILED"
# 应无输出

# 2. 确认可执行文件架构正确
file install/mpc_controller/lib/mpc_controller/mpc_controller_node
# 应包含 "aarch64" 或 "ARM aarch64"

# 3. 确认无 x86 残留库
find install/ -name "*.so" -exec file {} \; | grep x86
# 应无输出
```

### 9.2 运行时检查

```bash
# 1. 启动 FAST_LIO，确认 /Odometry 发布
ros2 topic hz /Odometry
# 应有输出，频率约 200Hz

# 2. 启动地形分析器，确认地图发布
ros2 topic hz /car_0_terrain_analyzer/fused_map
ros2 topic hz /car_0_terrain_analyzer/sdf_map
# 应有输出

# 3. 启动 MPC，确认能求解
ros2 topic echo /hunter_se/cmd_vel
# 发送目标点后应有控制输出

# 4. 确认 CPU 使用率合理
top -b -n 1 | grep -E "mpc|fast_lio|terrain"
# 不应持续 100%，留有余量
```

### 9.3 性能基准

| 指标 | 仿真 (x86) | ARM 目标 | 说明 |
|------|------------|----------|------|
| FAST_LIO 频率 | 200Hz | ≥100Hz | 低于 50Hz 会丢帧 |
| 地形分析频率 | 15Hz | ≥10Hz | 低于 5Hz 会延迟 |
| MPC 求解时间 | ~10ms | ≤40ms | 必须小于控制周期 |
| 规划时间 | ~300ms | ≤500ms | 含前端+后端优化 |
| CPU 总占用 | ~40% | ≤80% | 留余量给系统 |

---

## 十、移植工作流程总结

```
步骤 1: ARM 平台装系统
    └─ Ubuntu 22.04 aarch64

步骤 2: 装 ROS2 Humble
    └─ ros-humble-desktop

步骤 3: 装系统依赖
    └─ Eigen3, PCL, OpenCV, Boost, Protobuf, Qt5

步骤 4: 装 CasADi
    └─ pip3 install casadi 或源码编译

步骤 5: 装 Livox SDK
    └─ 源码编译 Livox-SDK2，安装到 /usr/local

步骤 6: 复制源码
    └─ scp -r seb_clean/src/ arm-board:~/seb_clean/src/
    └─ 不要复制 build/ install/ log/

步骤 7: 修改源码
    └─ ekf_quat_pose: 删除 -march=native
    └─ FAST_LIO: 修改架构判断，启用 ARM 多线程
    └─ Gazebo 仿真包: 修改 include 路径（实车可跳过）

步骤 8: 编译
    └─ colcon build --symlink-install

步骤 9: 修改参数
    └─ MPC: 轴距、速度、转向角
    └─ 地形: 障碍物高度、台阶阈值
    └─ 规划: 速度、安全裕度

步骤 10: 验证
    └─ 按检查清单逐项验证

步骤 11: 实车调试
    └─ 先低速、小范围测试
    └─ 逐步放开速度和范围
```

---

## 附录 A：依赖包完整列表

### 系统级依赖（apt）

```
libeigen3-dev libpcl-dev libopencv-dev libboost-all-dev
libprotobuf-dev protobuf-compiler libomp-dev qtbase5-dev
python3-dev python3-pip python3-venv cmake gfortran swig
```

### ROS2 依赖（apt）

```
ros-humble-desktop
ros-humble-tf2 ros-humble-tf2-ros ros-humble-tf2-geometry-msgs
ros-humble-pcl-ros ros-humble-pcl-conversions
ros-humble-cv-bridge ros-humble-message-filters
ros-humble-rosbag2 ros-humble-pluginlib
ros-humble-robot-state-publisher ros-humble-xacro
ros-humble-ackermann-msgs ros-humble-gazebo-ros-pkgs
```

### 外部依赖（源码/pip）

```
CasADi (pip3 install casadi 或源码编译)
Livox-SDK2 (源码编译，git clone)
```

## 附录 B：修改文件清单

| 文件 | 修改内容 | 必要性 |
|------|----------|--------|
| `src/tank_slam/ekf_quat_pose/CMakeLists.txt` | 删除 `-march=native` | 必须 |
| `src/FAST_LIO/CMakeLists.txt` | 修改 x86 架构判断为通用判断 | 必须（性能） |
| `src/simulator/livox_laser_simulation_ros2/CMakeLists.txt` | Gazebo include 改为动态查找 | 仿真必须 |
| `src/mpc_controller/params/controller_hunter_gazebo.yaml` | 轴距/速度/转向角适配 | 实车必须 |
| `src/uneven_mapping/terrain_analyzer/params/hunter_fastlio_registered_live.yaml` | 障碍物阈值适配 | 实车必须 |
| `src/planner/arc_planner/plan_manage/config/planning_hunter_gazebo.yaml` | 速度/安全裕度适配 | 实车必须 |
| Launch 文件 | cmd_topic 参数传递 | 实车必须 |
