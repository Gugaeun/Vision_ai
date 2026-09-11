# vision_ai

Vision AI 실습 프로젝트 모음 저장소입니다. 프로젝트별로 폴더가 나뉘어 있고, 각 폴더 안에 자세한 README가 있습니다.

## 폴더 구성

| 폴더 | 설명 |
|---|---|
| [`pen_detection/`](pen_detection) | YOLOv8을 커스텀 학습시켜 볼펜(pen) 1종을 탐지하는 프로젝트 |
| [`Requirements_DINO/`](Requirements_DINO) | 냉장 진열대 이상(상한 과일, 곰팡이, 포장 훼손 등) 탐지 데모 — Grounding DINO 구현 |
| [`Requirements_YOLO/`](Requirements_YOLO) | 위와 같은 진열대 이상 탐지 데모 — YOLO-World 구현 |
| [`Requirements_YOLOE/`](Requirements_YOLOE) | 위와 같은 진열대 이상 탐지 데모 — YOLOE 구현 |
| [`HumanEmojiConverter/`](HumanEmojiConverter) | YOLO-Pose + XGBoost로 자세(턱 괴기/양손 들기/양손 얼굴 가리기)를 분류해 실시간으로 이모지 얼굴을 띄우는 데모 |
| [`CrowdDemographics/`](CrowdDemographics) | DeepFace로 군중 영상 속 얼굴들의 나이·성별·감정을 집계해 인구통계 리포트를 만드는 데모 |

`Requirements_DINO`/`Requirements_YOLO`/`Requirements_YOLOE` 세 구현을 같은 사진·프롬프트로 비교한 내용은 [`docs/comparison.md`](docs/comparison.md)에 정리해뒀습니다.
