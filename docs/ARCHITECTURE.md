# Architecture

## 设计结论

系统采用“事实先行、语义后置”的分层管线。Body Pose 保持原
`models/yolov8n-pose.onnx` COCO-17 链路，Hand Pose 是独立、可关闭且失败
不阻塞身体链路的证据层。Pose 层保留检测、预测、插值和缺失状态；零件层只
接收真实模型输出；交互层只能产生 `derived_interaction_candidate`；工序层
必须引用上游事件 ID 并保持 `proposed`、`uncertain`、`not_observed` 或
`unavailable`。

```text
VideoFileSource (original-resolution frame)
  |
  +--> Existing YOLO Body Pose --> anonymous candidates
  |          |                         |
  |          v                         v
  |     validator/smoother ----> AnonymousPersonLock
  |          |                  person_ref / lock_epoch
  |          +--> pose frame evidence
  |          |
  |          +--> shoulder/elbow/wrist + original frame
  |                        |
  |                        v
  |             dynamic left/right hand ROI
  |                        |
  |                        v
  | DisabledHandBackend | MediaPipeHandLandmarkerBackend
  |                        |
  |                        v
  |       hand_pose_frames (detected/uncertain/missing)
  |
  v
pose_segments --> confirmed/hysteretic stable action_events
  |                    |
  |                    +-------------------------------+
  v                                                    |
ObjectPerceptionProvider                               |
  | real object_tracks or unavailable                  |
  v                                                    |
InteractionFusionEngine <------------------------------+
  | derived evidence only
  v
TemporalActionModel (interface; real model not configured)
  |
  v
ProcessReasoner
  |
  v
Web API + original timeline UI
```

## Body Pose 与 Hand Pose 后端

- `BodyPoseBackend` 保持现有 YOLO ONNX 检测、COCO-17身体关键点、匿名人物
  候选和锁定基础，不覆盖或重新训练原模型。
- `HandPoseBackend` 是旁路协议；`DisabledHandBackend` 提供诚实关闭状态，
  `MediaPipeHandLandmarkerBackend` 使用官方 Hand Landmarker `float16/1`
  与项目本地 MediaPipe `0.10.35`。
- 手部 ROI 由同一锁定人物的肩、肘、腕和原分辨率画面产生。关键点从裁剪
  坐标映射回原视频坐标，`crop_bbox` 与 `crop_transform` 必须可追溯。
- 左右侧以 Body Pose 的解剖学左右为主约束，手模型 handedness 只能用于
  一致性检查，不能覆盖 `anatomical_side`。
- 成功检测才允许输出21个真实点。低质量/不可见输出 `uncertain` 或
  `missing`；missing 的 landmarks 为空，不生成固定手型或 mock 手指。
- Hand backend 的关闭、未配置或单帧失败不得终止视频解码、Body Pose、锁定
  或动作稳定化。

模型来源、许可证、运行时和文件哈希见 `HAND_MODEL_MANIFEST.json`；
选型边界与当前验证状态见 `docs/HAND_POSE_TECHNICAL_DECISION.md`。

## 动作双层结构与 Phase B 稳定化

`pose_segments` 保留短时底层观测和来源证据；Web 主时间线只使用
`action_events`。每个稳定事件必须引用完整的 `source_segment_ids`。

当前 Phase B 配置为：

- `analysis_fps=8.0`
- `stable_event_minimum_seconds=1.2`
- `short_directional_event_minimum_seconds=1.0`
- `short_gap_merge_seconds=0.4`
- `start_confirmation_seconds=0.5`
- `stop_confirmation_seconds=0.5`
- `temporal_context_seconds=2.5`

开始确认、停止迟滞、短间隔合并和上下文只作用于真实观测形成的名称与边界，
不得机械延长时间。小于门限且不能可靠合并的片段只能在稳定层被抑制，或保留
为 `transition`/`unknown`；没有物体证据和人工确认时，不为 release 等微动作
开放不足1秒的语义例外。

## 包边界

- `src/video_io`：中文/空格路径、元数据、抽帧和哈希。
- `src/pose_core`：真实 YOLO ONNX 推理、骨架、状态感知渲染。
- `src/hand_pose`：可插拔后端、原图动态 ROI、坐标回映、左右绑定与手部状态。
- `src/tracking`：匿名锁定、人物候选、显式换人和人工 relock 边界。
- `src/action_segmentation`：Pose-only 粗动作、底层分段、确认/迟滞、上下文和
  稳定事件。
- `src/object_perception`：真实零件模型协议；当前为 not-configured provider。
- `src/interaction_fusion`：人体/手部与零件的派生候选接口，禁止声称真实抓握。
- `src/temporal_actions`：数秒上下文模型协议；当前无训练模型。
- `src/process_reasoning`：证据门控工序步骤；证据不足时不生成步骤。
- `src/web`：原创 Web 服务、Range 视频、状态 API、稳定时间线和 Canvas
  Body/Hand Pose。
- `src/legacy_pose`：按来源文件保留的最小稳定技术底座。

## 人物、手部与事件硬边界

人物仅在一次分析会话内使用匿名 `person_ref`。`lock_epoch` 表示一次明确锁定
范围。跟踪器若选择不同 track，包装层将变化暴露为
`awaiting_manual_relock` 并停止正常动作输出，直到显式确认。身体点、手部点
和稳定动作均不得跨：

- `person_ref` 或人物切换；
- `lock_epoch` 或 manual relock；
- lost/off-frame；
- severe occlusion；
- long missing。

手部绑定还包含 `anatomical_side`，不得把左/右手跨侧合并。lost 时不能继续
输出正常手部几何或正常动作；恢复后的状态必须由新观测重新确认。

## 资源与进程

离线 Web worker 保留 spawn-safe 入口，ONNX Runtime 与 MediaPipe 后端在
受控进程内构造。当前只运行离线 Video Analysis；Camera 入口在 UI 中明确为
unavailable，不运行 USB 或 RTSP。未来 Camera 模式必须复用单一资源所有权和
有界队列。

MediaPipe 本地烟雾运行中观察到 Clearcut 性能/使用指标网络尝试，受限环境已
阻断。部署前必须审查遥测字段、禁用机制和出口策略；在此之前建议保持执行
环境禁止外联。

## Web 状态

- `/health`：轻量可用性。
- `/api/status`：身体/手部模型、数据、计数、锁定和四个验证标志。
- `/api/analysis`：本次真实身体、手部、动作与下游证据。
- `/media/video`：支持200、206和非法 Range 响应的真实视频。

前端按视频当前时间选择对应的真实抽样 Body Pose 与 Hand Pose；Body/Hand
分别可切换，左右手采用不同颜色。当前稳定动作与主时间线来自
`action_events`，底层 `pose_segments` 只在来源证据中展开。lost 或 hand
missing 时保留视频画面，但移除不可靠几何并显示状态。

专项验证使用3个各12秒的真实片段。升级前2 fps受控重放共72帧、42个
`pose_segments`、11个动作事件记录（其中4个正常稳定动作），不足1秒稳定动作1个；
升级后8 fps共288帧、120个底层片段、11个动作事件记录（其中2个正常稳定动作），
不足1秒稳定事件为0。更高采样率使底层片段和抑制计数不可直接按绝对数量与旧
采样率比较；主时间线事件总数未增加。

升级后执行266次手部推理：帧状态 detected/uncertain/missing 为
7/34/247，左右侧观测为7/48/521。39条侧级警告中的25个唯一帧级关联错误候选
被降级；lost期间正常动作误报为0，跨人物或epoch合并为0。底层稳定化证据的
unknown/transition为14.999351→29.116271秒，主时间线显示值为
11.960836→9.912747秒；真实运行的 `merged_fragment_count=0`，短间隔
合并能力只由专项合成边界测试证明，不能声称在这3段真实片中实际发生了合并。
平均 Body Pose/Hand Pose 推理时间分别为36.633/18.207376 ms，端到端速度
为6.698565 fps。

这些是运行与时间线稳定性证据。没有独立人工真值，Macro-F1、关键点准确率和
左右关联准确率均为 `not_evaluable`；7个 detected 帧尚未逐帧人工确认，且
247/288帧为 missing，当前机位的真实手部覆盖不足。

## Phase B.1 双时间轨与观测支持架构

Phase B.1 保持原始证据和稳定语义分层，不用稳定标签覆盖 Pose 事实：

```text
pose_frames
   |
   +--> raw pose_segments ------------------------------+
   |                                                    |
   |                                                    v
   |                                      evidence_timeline
   |                                 100% analysis-window coverage
   |                         normal / transition / unknown /
   |                              uncertain / lost
   |
   +--> per-side temporal lanes
           left / right / bilateral
           start confirmation / stop hysteresis
           bounded_uncertain_gap <= 0.375 s
           person / epoch / lost / long-missing hard reset
                  |
                  v
        pre-gate same-lane support aggregation
        observed support excludes gap and other-side evidence
                  |
                  v
             action_events
       stable actions + explicit hard boundaries
       source_segment_ids + bounded gap lineage
```

Web 将两层作为独立轨道：

- 连续证据轨只读 `evidence_timeline`；旧产物缺失该字段时，界面会明确标记为
  `pose_segments fallback`，不会伪装成已发布的完整证据轨。
- 稳定动作轨只读 `action_events`；短 lost/hard boundary 即使不是正常稳定动作，
  仍显示为边界证据。
- 三条轨道（工序、连续证据、稳定动作）统一使用分析窗口相对几何，但点击后跳到
  完整视频的绝对时间。
- 视频当前帧、Body/Hand overlay、当前证据 chip、稳定事件和来源片段使用同一个
  绝对时间轴。
- 浏览器主动取消 Range 请求属于预期客户端断连；服务器只静默处理
  `BrokenPipeError`、`ConnectionAbortedError` 和 `ConnectionResetError`，
  其他异常继续暴露给专项测试。

Phase B.1 没有改变 Hand 模型、Object/Interaction/Process 空状态，也没有把 Hand
点用于动作命名。四个验证标志保持 `false`。

## 经典 Body Pose Web Renderer

默认 Web Body Skin 使用独立的
`src/web/static/classic_body_pose_renderer.js`。它把当前真实 COCO-17
关键点映射为青色主体骨架，并把头圈、颈部、髋中心、身体中线以及腕部/足部视觉延伸
显式标记为 `derived_visual_only` 的洋红色几何。派生几何只存在于 Canvas 绘制过程，
不会写回 `analysis.json` 或替代原始关键点。

`detected`、`predicted`、`interpolated` 和 `derived_visual_only` 分别使用不同的
颜色、透明度和线型。`missing`、`uncertain`、`rejected`、`lost` 不生成固定几何。
经典模式与原绿色“证据骨架”模式为互斥分支，Body Pose 和真实 Hand Pose 则继续作为
独立图层开关。该层不改变人物锁定、动作分段、时间线或 Web API 契约。

## Local USB Camera technical path

The optional Camera mode is a local-only technical adapter:

`Camera UI → start/status/stop API → Windows spawn worker → OpenCV USB source → real YOLO Body Pose → anonymous lock → real Hand backend → conservative live action state → latest-frame JPEG + evidence → classic Canvas renderer`

- A single `AnalysisResourceCoordinator` lease prevents Camera and offline
  analysis from owning inference resources together.
- Transport queues are bounded and latest-frame-only. Backpressure may drop
  transport frames, but it never creates Pose or Hand evidence.
- The worker uses monotonic timestamps, explicit cancellation, a bounded stop
  timeout and device release in `finally`.
- Camera recording is disabled. RTSP, PLC and MES are outside this adapter.
- Live actions remain `proposed`/`uncertain`, with
  `training_eligible=false`. Missing Hand evidence contains no geometry.
- A local laptop Camera check does not change
  `factory_camera_validated=false`.

## Temporal Action Engine V3 shadow path

`src/temporal_actions/engine_v3.py` implements the first executable V3
interpretable-rules provider. It remains a shadow layer:

```text
accepted pose_frames + pose_segments + action_events
    -> 2.5-second feature buffers
       partitioned by person_ref / lock_epoch / left-right-bilateral lane
    -> real Body motion features
    -> optional qualified-only real Hand motion/shape features
    -> Object features unavailable
    -> traceable shadow candidates + change-point evidence
    -> accepted Phase B action_events remain the primary Web timeline
```

Every source Body frame produces independent left, right and bilateral feature
records. Lost/off-frame, manual-relock, explicit hard-boundary, person and epoch
changes clear only the affected temporal partition. Missing or unqualified Hand
records carry no Hand geometry into the action feature layer. A Hand record is
eligible only when it contains 21 real landmarks, is bound to the same
`person_ref`, `lock_epoch` and anatomical side, passes the quality gate, and has
no duplicate-side or association warning.

The Object feature state is `unavailable`; no Object track or Interaction ID is
created. Object-dependent actions are conservatively mapped to a Pose-only
fallback before a shadow feature can validate. V3 provider errors return
`unavailable` and leave the accepted action timeline available. Automatic
records remain `proposed` or `uncertain`, with `training_eligible=false`.

The authoritative three-window shadow A/B retained all four normal stable
actions, emitted no normal event shorter than one second, retained 100%
continuous-evidence coverage, and produced zero lost false positives and zero
cross-person/epoch merges. Because there is no human ground truth and the
traceable shadow support span is slightly shorter than the accepted event span,
V3 was accepted as shadow infrastructure but was not promoted to the primary
timeline.

## GPU Body Pose and low-latency display path

Video and Camera now share the provider selector in `src/pose_core/providers.py`.
The supported policies are `auto`, `prefer_cuda`, `require_cuda`, and `cpu`.
The reported active provider comes from the created session's
`session.get_providers()` result. `require_cuda` fails closed; only
`prefer_cuda` may fall back, and its reason is exposed to the API and UI.
The unchanged `yolov8n-pose.onnx` Body model runs on
`CUDAExecutionProvider` in the accepted GPU path. MediaPipe Hand remains an
independent CPU backend and is labelled CPU.

Camera transport keeps a bounded exact-sequence snapshot cache. The atomic
packet endpoint contains the JPEG and evidence from one sequence; an evicted
or unknown requested sequence returns a stale-sequence error and is never
replaced by a newer JPEG. The browser uses latest-only scheduling and sends an
explicit display acknowledgement so frame age, dropped display frames, and
sequence mismatches can be measured.

Live cadence is deliberately split:

```text
USB capture (requested 15 FPS)
    -> Body Pose / classic overlay display target (12 FPS)
       -> latest-only atomic JPEG + evidence
    -> conservative action sampling target (8 FPS)
       -> start confirmation / hysteresis / existing boundary rules
```

Display-only frames do not append action evidence. They may hold the most
recent stable action for presentation only, with `action_sampled=false`,
`held_for_display=true`, and the original source action frame preserved.
Identity, epoch, lost, or lock-state changes force an action sample so a display
cadence split cannot bridge a hard boundary.

Offline Video overlays are driven by `requestVideoFrameCallback`; environments
without it use a bounded `requestAnimationFrame` fallback. `timeupdate` remains
only a compatibility signal. Seek, event jump, pause, and playback-rate changes
therefore request an immediate overlay refresh without changing stored Pose or
action evidence.

## Local MP4 intake and cancellable analysis jobs

The local-only intake path is:

```text
browser file input
  -> streamed POST body into a unique .part file
  -> maximum-size gate + SHA256 + real codec/decode probe
  -> atomic source.mp4 promotion
  -> start-time real Body preview
  -> opaque initial-candidate token + explicit user selection
  -> Windows spawn offline worker
  -> CUDA Body + CPU Hand + accepted tracking/action pipeline
  -> unique outputs/analyses/analysis_<uuid>
  -> hot activation in the existing Range player
```

The UI never sends a filesystem destination or trusts a client bbox as the
worker lock. The server resolves the upload ID inside the controlled intake
root and the worker re-runs real Body detection to match the manual seed.
Cancellation is checked inside the frame loop. A worker error releases the
shared Pose lease and exposes a safe summary to the UI while retaining local
diagnostics.

## Selectable Camera manual relock

Each real Camera candidate is projected through two contracts:

1. The worker keeps the real Body detection, torso evidence, source frame,
   dimensions, mirror state and fingerprint in a bounded private history.
2. The controller publishes only an opaque token plus anonymous display fields.
   The token is bound to the current Camera session and exact transport
   sequence, expires after four seconds, and is rejected outside a bounded
   recent-sequence window.

The user enters selection mode, clicks a displayed candidate box, and then
explicitly confirms or cancels. Overlapping boxes use the smallest containing
box rule. Confirmation cannot submit a raw `track_id`; the worker must find the
token's exact candidate fingerprint in its private history. A successful
selection rebuilds the manual Body tracker, so validator and smoother history
cannot cross the person boundary, resets the Hand backend context, clears the
causal action frames and deadlines, and establishes a new `person_ref` and
incremented `lock_epoch` only after the next real detection matches. Cancellation
only cancels the UI selection and never changes a person.

If the selected candidate expires, disappears, belongs to another session or
fails worker revalidation, the system stays lost/uncertain and requires a new
explicit selection. No automatic highest-confidence Relock is performed.
