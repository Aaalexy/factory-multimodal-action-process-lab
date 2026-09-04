# Temporal Action Engine V3

## 当前状态

Temporal Action Engine V3 处于可审计的 `shadow_mode`。Phase C.0 只冻结
特征帧、动作候选和 provider 的契约；它不接管现有 `action_events`，不改变
Phase B.1 的连续证据时间线或稳定动作时间线，也不把 Hand 或 Object 证据用于
动作命名。

当前主时间线仍来自 Body Pose 启发式分类、开始确认、停止迟滞、边界门控和稳定
事件聚合。学习式 Temporal provider 未配置时必须返回 `unavailable` 和空候选。

## 契约层

`temporal_feature_frames` 为逐采样时刻的三通道特征证据预留结构：

- `left`、`right`、`bilateral` 分别绑定 `person_ref` 和 `lock_epoch`；
- 保留原始帧、Pose segment、Hand、Object 和 Interaction 来源 ID；
- 记录规则版本、参数 profile ID 及参数载荷 SHA256；
- 分开记录 Body、Hand、Object 的状态和“是否实际使用”；
- 硬边界必须是 `uncertain` 或 `lost`，不得使用 Hand/Object 特征；
- 自动输出只能是 `proposed` 或 `uncertain`，且
  `training_eligible=false`。

`temporal_action_candidates` 为未来可解释规则引擎的候选输出：

- 只允许项目既定动作词表；
- 要求正时间跨度、帧范围、来源特征和 Pose segment lineage；
- change-point 必须引用候选自身的 Temporal feature；
- candidate 必须与所有来源 feature 的人物、epoch、解剖侧、视频、录制组、
  规则版本和参数 profile 完全一致；
- candidate 的帧索引与时间端点必须由引用 feature 精确支撑，不得机械扩展；
- `carry`、`place`、`hold`、`release` 必须引用真实 Object track 和
  Interaction evidence；
- `transition`、`unknown`、`lost` 必须保持 `uncertain`；
- `lost` 必须是硬边界，不能跨人物或 epoch 延续；
- 所有候选保持 `shadow_mode=true`，不能被误认为当前主动作或生产结论。

权威字段定义见：

- `schemas/temporal_feature_frames.schema.json`
- `schemas/temporal_action_candidates.schema.json`
- `configs/temporal_action_v3.json`
- `src/temporal_actions/contracts.py`

## Hand 与 Object 门

Hand 只有同时满足以下条件时才可标记为 `hand_features_used=true`：

1. `hand_feature_state=qualified`；
2. 有真实 `source_hand_pose_ids`；
3. 特征来自相同 `person_ref`、`lock_epoch` 和 `anatomical_side`；
4. 未来运行时再次核验源 Hand record 的
   `action_feature_eligible=true`。

这类特征最多表示 `hand_motion_feature` 或 `hand_shape_feature`，不等于真实抓握、
拿取完成或装配完成。

当前 Object provider 未配置，因此 `object_feature_state=unavailable`，
`object_features_used=false`，不得产生 Object 语义动作或 Interaction。

Provider 输出校验还要求所有 Pose segment、Hand、Object 和 Interaction ID
存在于调用方提供的上游 ID 集合中。仅填写格式正确但不存在的 ID 不构成真实
lineage，输出会 fail closed。

## Phase C.0 验证边界

冻结的三个真实窗口共包含：

- 288 个 Body Pose 帧；
- 576 条 Hand 记录；
- 120 个 `pose_segments`；
- 13 个既有 `action_events`；
- 94 个 `evidence_timeline` 区间。

Phase C.0 只验证契约、Schema、空 provider、来源哈希和现有 Web/Range 回归。
没有人工动作真值，因此 Macro-F1、precision 和 recall 仍为
`not_evaluable`。

下一独立 checkpoint 才会实现 left/right/bilateral 的 2.5 秒数值特征缓冲。
该缓冲仍先以 shadow 方式运行；只有同三段真实窗口的边界、安全、覆盖、性能和
来源追溯门全部通过后，才会评估 change-point 或 V3 候选动作。
