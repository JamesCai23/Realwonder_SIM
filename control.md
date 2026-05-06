# RealWonder 仿真控制指南

本文档介绍了如何控制 `/home/lff/data1/cym/physical_data/rw_sim` 系统中物体的仿真位置、速度以及可见性。

## 1. 核心控制文件
所有的控制参数都在输入 CSV 文件中定义，或者是通过 `config.yaml` 传递给 `genesis_simulator.py`。

## 2. 运动控制

### 运动模式 (`motion_mode`)
- `constant_velocity`: 匀速运动。初始速度由 `initial_velocity` 决定（默认标量为 2.0）。
- `constant_acceleration`: 匀加速运动。由 `force_direction` 和 `force_strength` 决定推力。
- `free_fall`: 自由落体。受 `gravity` 影响（默认 -9.8）。

### 速度与力
- **初速度 (`initial_velocity`)**: 格式为 `[vx, vy, vz, wx, wy, wz]`。
  - *注意*: 视野 X 轴对应 Genesis 的 X 轴，视野 Y 轴对应 Genesis 的 Z 轴（垂直方向）。
- **推力强度 (`force_strength`)**: 控制匀加速模式下的加速度大小。

## 3. 视野 (FOV) 与位置控制

### 视野范围 (以距离 d=1.0 为例)
- **水平 (X)**: 约 $\pm 0.7$ 单元。
- **垂直 (Z)**: 约 $\pm 0.4$ 单元。

### 保证物体在视野内
1. **控制位移**: 
   - 匀速模式建议速度 $v \le 0.5$。
   - 匀加速模式建议强度 $a \le 1.0$。
2. **初始偏移 (`initial_position_offset`)**: 
   - 可以在 `config.yaml` 中设置 `[dx, dy, dz]` 来调整物体的起始位置。
3. **坐标映射**:
   - `+X`: 向右移动
   - `-X`: 向左移动
   - `+Z`: 向上移动
   - `-Z`: 向下移动
   - `+Y`: 远离相机（在 Genesis 坐标系中 Y 为深度方向）

## 4. 仿真时间控制
- **帧数 (`simulated_frames_num`)**: 默认为 51 帧。
- **步长 (`dt`)**: 默认为 0.02s。
- **总时间**: `simulated_frames_num * dt`。增加帧数会线性增加运动时间。

