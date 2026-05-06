# Realwonder-SimOnly
只需要完成 T2I, I2S 并导出仿真视频(final_sim/simulation.mp4 及其 相关运动数据), 不需要实现 Controlled Wan Model to Video.

## Text-to-Image
复用现有链路, 直接调用 vllm-based Image Generation 即可
- Endpoint: 8000 (已经部署好)
- Qwen-Image 模型

## Image-to-Scene/Simulation
复用现有链路, 但需要对于Batch Inference实现 Model 持久 Loaded, 即持续在GPU上加载好模型以供批量推理

## Save Artifacts
保存逻辑和现有的一致, 保存仿真视频、运动学数据

## TODO
在 /scripts 下实现
`batch_simulation.sh`:
- 根据给出的csv(e.g. /home/lff/data1/cym/physical_data/rw_sim/scripts/test_mini.csv)实现批量仿真
- 你可以创建新的python脚本来实现这一目的

验证正确性, 在 conda env `realwonder` 下运行并实现所有仿真视频以及动力学信息, 最后用/home/lff/data1/cym/physical_data/rw_sim/generate_quantiphy_qa.py实现QA pairs生成.
- 现在的QA逻辑没有物体名称, 只有 `object_0` 这样的占位, 需要予以修复.
- 对于一个视频, prior 只能有一个, 如果第一个问题是length作为prior, 那么之后的所有问题也只能是length作为prior, 否则会造成泄漏.

**要点**: 加载一次, 多次推理, GPU 指定正确(自行查找可用GPU), 结果规范.


## 数据过滤
- 对于 Inpaint 有问题的图片(inpaint后和原图一致) /home/lff/bigdata1/cym/realwonder_simdata/result/balloon_rise_01
- simulation.mp4静态 /home/lff/bigdata1/cym/realwonder_simdata/result/balloon_rise_01
- optical_flow 问题 /home/lff/bigdata1/cym/realwonder_simdata/result/balloon_rise_01

需要过滤掉避免影响数据质量