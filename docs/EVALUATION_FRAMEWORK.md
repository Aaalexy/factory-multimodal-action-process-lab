# Evaluation Framework

没有人工真值时，脚本输出 `not_evaluable`，所有指标值为null。不得用模型预测
互相对比来代替人工真值。

## 动作层

- Macro-F1
- per-class precision / recall / F1
- 动作边界误差（秒）
- 小于1秒稳定事件数量
- 每分钟事件数量
- 过度分割次数
- lost/off-frame期间动作误报
- 人物切换次数

## 零件层

- object mAP
- per-class precision / recall
- object track continuity
- 遮挡恢复错误

## 交互层

- interaction precision / recall / F1
- 解剖学左右侧错误率
- 错误关联人物或零件的数量

## 工序层

- process step accuracy
- 工序边界误差
- 顺序错误率
- 重复步骤识别
- unknown/uncertain覆盖率
- 错误生产结论数量，目标固定为0

入口为 `scripts/evaluate.py`。后续人工真值格式确定后，应在同一入口增加按录制
组评估与置信区间，而不是改变指标定义。
