# Review Interface Feature Matrix

This public matrix describes code-level capabilities. It is not a production-deployment or visual-acceptance claim.

| Area | Current implementation | Evidence boundary |
| --- | --- | --- |
| Video intake | Local authorized video is copied into ignored output storage | No private video is distributed |
| Body pose | COCO-17 keypoints, provider status, anonymous person lock | Wrist keypoints are not detailed hand pose |
| Hand landmarks | Optional MediaPipe evidence with missing/uncertain states | Does not establish grasp or object contact |
| Action timeline | Pose-derived segments and stabilized candidate events | Candidates require human review |
| Object perception | Explicit `not_configured` state | No substitute object labels |
| Interaction fusion | Explicit unavailable state without upstream evidence | No inferred contact |
| Process reasoning | Explicit unavailable state without supporting models/evidence | No automated process decision |
| Camera path | Local USB lifecycle and resource coordination | Hardware-specific checks require an authorized device |
| Review UI | Local browser view of overlays, warnings, and timelines | Static/API tests are not visual acceptance |

Public/core tests cover source-controlled contracts and synthetic fixtures. Hardware, private-video, and generated-evidence regressions are marked `private_artifacts`.
