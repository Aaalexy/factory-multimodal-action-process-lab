# Data Governance

## 自动结果不是人工真值

任何自动分析、模型输出或AI辅助语义默认：

- `status=proposed` 或 `status=uncertain`；
- `training_eligible=false`；
- `training_approval=pending`；
- `reviewer=null`；
- `reviewed_at=null`。

系统不得自动写入 `confirmed`、`training_eligible=true` 或
`human_confirmed_semantic`。只有包含明确审核人、审核时间、审核决定和来源
证据的记录才有资格进入后续人工真值流程。

## 必须保留的溯源字段

- `source_video_sha256`
- `recording_group_id`
- `person_ref`
- `lock_epoch`
- `source_model_version`
- `reviewer`
- `reviewed_at`
- `training_approval`

`SOURCE_IMPORT_MANIFEST.json` 记录迁移文件的来源路径、目标路径、来源SHA256和
最终目标SHA256。运行产物以输入视频SHA256和模型SHA256关联，不以文件名作为
唯一标识。

## 录制组与数据泄漏

- sample_set_A与sample_set_B是同一次连续录制、同机位，必须属于
  `recording_group_sample_ab`。
- sample_set_C是另一段录制且机位位于右侧，属于
  `recording_group_sample_c`。
- 不允许按帧随机拆分train/dev/test。
- 在至少第三个独立录制组出现前，`test` 状态为 `not_available`。
- 本轮不训练模型，所有短片保持 `split=unassigned`。

## 隐私

允许匿名人物候选、会话内锁定、Pose与动作技术证据。禁止人脸、姓名、工号、
服装ReID、跨视频员工身份、员工表现评分和由动作推断个人结论。

## 证据分层

1. Pose事实：关键点、人物框、检测/预测/插值/缺失状态。
2. 零件事实：只允许真实版本化物体模型输出。
3. 派生交互：腕点与零件框的时空关系，必须标记
   `derived_interaction_candidate`。
4. 时序动作：必须保留上下文窗口和来源事件。
5. 工序语义：必须引用动作/交互ID并经过人工复核。

缺少某层证据时返回 `not_configured`、`unavailable` 或 `not_observed`，不得
用固定框、固定骨架或预设工序补齐。

## Hand Pose 事实与自动结果

Hand Pose 是 Body Pose 之外的独立证据层，不得把手部语义混入 COCO-17
身体事实。每条 `hand_pose_frames` 自动记录至少保留：

- `hand_pose_id`
- `person_ref`
- `lock_epoch`
- `anatomical_side`
- `frame_index` 与 `timestamp`
- `crop_bbox` 与 `crop_transform`
- `landmarks`、`landmark_count` 与可取得的真实置信证据
- `observation_state` 与 `occlusion`
- `source_video_sha256`
- `recording_group_id`
- `source_model_version`
- `status`
- `reviewer`、`reviewed_at`
- `training_approval` 与 `training_eligible`

自动手部记录必须为 `status=proposed` 或 `status=uncertain`，
`training_eligible=false`、`training_approval=pending`、
`reviewer=null`、`reviewed_at=null`。模型成功返回的21点才可记录为真实
landmarks；`missing` 必须为0点和空数组，不能用固定手型、mock 点、历史点或
身体腕点补齐。

短时平滑、预测或插值若在未来启用，必须保留与 `detected` 不同的观测状态，
不得把派生点重新标成模型检测点。MediaPipe 未直接暴露的内部检测、存在或
跟踪置信分数必须为 `null`/`unavailable`，不得用 handedness 或其他分数
代填。

## 解剖学左右与绑定边界

`anatomical_side` 由同一锁定人物的 Body Pose 左右关系决定。手模型的
handedness 只用于一致性检查，不得覆盖身体左右。手部点必须同时绑定
`person_ref`、`lock_epoch` 和 `anatomical_side`，且不得跨以下边界复用、
平滑或合并：

- 人物变化或人工重新锁定；
- `lock_epoch` 变化；
- lost/off-frame；
- severe occlusion；
- long missing。

左右关联错误必须作为技术复核项记录，不能静默交换左右手。手部缺失不得阻塞
身体 Pose 或被解释为“没有动作”；它只表示当前没有足够的真实手部证据。

## 模型资产与依赖溯源

手部模型只允许存放在 `models/hand_pose`。当前单一资产的官方 URL、项目名、
版本、Apache License 2.0 许可证依据、文件大小、SHA256、下载时间、下载次数
和本地运行时版本记录在 `HAND_MODEL_MANIFEST.json`。本轮没有训练或微调
模型，也没有修改旧项目环境或系统 Python。

模型可加载不等于真实工位有效。CPU 空白图推理成功只证明运行链可调用。
Phase B 的3个真实12秒片段共288帧，帧级
detected/uncertain/missing 为7/34/247；576条左右侧观测为
7/48/521。266次手部推理的平均耗时为18.207376 ms。39条侧级警告中的25个
唯一帧级关联错误候选被门控降级，不得将其计入 detected。

上述计数是运行证据，不是人工真值。少量 detected 未经逐帧人工确认，左右
关联准确率、关键点准确率、遮挡恢复率和 Macro-F1 均为
`not_evaluable`。missing 占绝大多数，当前画面覆盖不足；这些结果不能作为
生产准入证据，也不能自动写入人工确认或训练资格。

## 手部证据的语义限制

21点手部几何不包含物体分割、接触力、夹持力或可靠的零件状态。手点、腕点和
零件框之间的时空接近只能生成
`derived_interaction_candidate`，不能直接写成真实抓握、拿取、放置、
装配完成或生产合格结论。零件模型未配置时，不得仅凭手部点生成零件或工序
事实。

## 运行时网络治理

本地 MediaPipe 烟雾运行中观察到向 Google Clearcut 发送性能/使用指标的网络
尝试，当前受限网络环境将其阻断。部署前必须完成依赖遥测、字段范围、禁用机制
和网络出口策略审查。完成审查前，建议在禁止外联的环境中运行，并把该项保持为
未通过的部署治理检查；不得由“模型在本地推理”推导出“运行时绝无外联”。

## 验证标志

Hand Pose 的加入不改变项目验证边界。以下标志继续保持 `false`：

- `factory_camera_validated`
- `production_action_model_ready`
- `external_factory_validated`
- `production_process_model_ready`

## 本地视频 intake 记录

每次浏览器上传必须保留独立 `upload_id`、原始显示文件名、受控存储路径、字节数、
SHA256、codec、分辨率、FPS、时长、可解码状态和上传时间。上传过程中的 `.part`
不是完整视频；只有大小限制和真实探测通过后才原子提升为 `source.mp4`。失败或中断
文件不得被作业控制器视为 ready，也不得用原始文件名覆盖已有 intake。

上传得到的 Body、Hand、动作及人工初始人物种子仍是自动技术证据：
`status=proposed` 或 `uncertain`，`training_eligible=false`。选择初始匿名人物只决定
分析目标，不等于确认动作、身份或训练真值。

## 人工 Relock 审计

Camera Relock 只能由显式用户操作触发。审计事件记录匿名 candidate ID、Camera
session、帧 sequence、操作类型、前后 `person_ref`/`lock_epoch` 和状态重置结果；
不记录姓名、工号、面部特征、服装特征或跨会话身份。

候选 token 是短期能力凭据，不是员工标识。它不得持久化为身份字段，也不得由客户端
替换为任意内部 track ID。确认后 Body、Hand 和动作上下文必须在新 epoch 上重新建立；
取消不得更改人物。所有 Relock 事件保持技术候选语义，不能自动授权训练，也不能产生
生产结论。

没有人工真值时，动作 Macro-F1、手部覆盖准确率、交互指标和工序指标均为
`not_evaluable`；不得把时间线更稳定或手点可见等技术结果改写为生产准确率。
