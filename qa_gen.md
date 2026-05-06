# RealWonder QA 生成逻辑总结

为了确保物理数值推理的合理性与准确性，`generate_quantiphy_qa.py` 遵循以下逻辑：

## 1. 物理尺度映射 (Physical Scaling)
仿真器使用通用单位运行。为了将其转换为真实世界的物理值：
- 系统从 `config.yaml` 中读取 `estimated_real_size` (来源于 CSV 的同名列)。支持显式带单位的输入，如 `8cm` 或 `0.2m`。
- 根据物体的仿真尺寸与估计真实尺寸，计算缩放因子 `scale = estimated_real_size / simulation_size`。
- 所有长度 (m)、速度 (m/s) 和加速度 (m/s^2) 都会乘以该因子进行转换。

## 2. 单位选择与题干格式
- **输入决定输出**: 在生成 QA 时，系统会遵循输入配置中的单位偏好：
  - 如果输入 `estimated_real_size` 使用 `cm`，则该视频的所有 QA 输出（Prior 和 Answer）及题干均强制使用 `cm`, `cm/s`, `cm/s^2`。
  - 如果输入使用 `m` 或未指定单位（默认），则使用 `m`, `m/s`, `m/s^2`。
  - 问题题干会明确包含目标单位，例如：`What is the length of the apple in cm?`。

## 3. Prior 选择与防泄漏
- **单 Prior 原则**: 每个视频会从 `['length', 'speed', 'accel']` 中**随机选择一个**作为唯一的 Prior 类型。
- 该视频的所有 QA 对都共享同一个 Prior，防止交叉泄露。

## 3. 有效性检查
- **非零检查**: Prior 和 Answer 必须大于有效阈值 (默认 `1e-3`)。
- 逻辑上避开 `0.0` 这种没有推理意义的数值，确保问题具有物理挑战性。

## 4. 任务类型映射
- **SD (Size to Dynamic)**: 给定长度 -> 推理速度/加速度。
- **DS (Dynamic to Size)**: 给定速度/加速度 -> 推理物体长度。
- **DD (Dynamic to Dynamic)**: 给定速度 -> 推理加速度 (或反之)。

## 5. 物体定位与命名
- 优先从 `all_object_names` 中获取真实名称。
- 如果没有名称，则尝试从 `case_name` 推断或使用 `object_{idx}` 占位。
