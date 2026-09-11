import sys

sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')

import cv2
from deepface import DeepFace
import os
import glob
import time
import csv

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
VIDEOS_DIR = './videos'
REPORTS_DIR = './reports'
os.makedirs(REPORTS_DIR, exist_ok=True)

BACKENDS = ['opencv', 'retinaface']  # 요건: 백엔드 2개 이상을 같은 조건(같은 영상)으로 비교
SAMPLE_EVERY_SEC = 1.0


def find_video():
    files = sorted(glob.glob(os.path.join(VIDEOS_DIR, '*.mp4')) +
                    glob.glob(os.path.join(VIDEOS_DIR, '*.avi')) +
                    glob.glob(os.path.join(VIDEOS_DIR, '*.mov')))
    return files[0] if files else None


video_path = sys.argv[1] if len(sys.argv) >= 2 else find_video()
if not video_path:
    print(f"Error: {VIDEOS_DIR}/ 에 분석할 영상이 없습니다.")
    exit()
print(f"비교 대상 영상: {video_path}")

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
sample_every_n = max(1, round(fps * SAMPLE_EVERY_SEC))

# 두 백엔드가 완전히 동일한 프레임을 보도록, 프레임을 먼저 메모리에 뽑아둔 뒤 비교함
frames = []
idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if idx % sample_every_n == 0:
        frames.append(frame)
    idx += 1
cap.release()
print(f"샘플 프레임 {len(frames)}개로 백엔드 {len(BACKENDS)}종을 같은 조건에서 비교합니다.")
print("(군중 영상은 얼굴이 작고 겹쳐 있어서, 백엔드마다 '몇 명을 찾아내는지' 자체가 크게 달라질 수 있습니다)\n")

summary = []
for backend in BACKENDS:
    total_faces = 0
    total_time = 0.0
    for frame in frames:
        t0 = time.time()
        try:
            results = DeepFace.analyze(frame, actions=['emotion'], detector_backend=backend,
                                        enforce_detection=False, silent=True)
            results = [r for r in results if r.get('face_confidence', 1) > 0]
        except Exception:
            results = []
        total_time += time.time() - t0
        total_faces += len(results)

    n = len(frames)
    avg_time = total_time / n if n else 0
    avg_faces = total_faces / n if n else 0
    summary.append({
        'backend': backend,
        'avg_faces': avg_faces,
        'total_faces': total_faces,
        'avg_time_ms': avg_time * 1000,
        'fps_equiv': 1 / avg_time if avg_time > 0 else 0,
    })
    print(f"[{backend:10s}] 프레임당 평균 검출 인원 {avg_faces:.1f}명 (총 {total_faces}건)  "
          f"프레임당 평균 {avg_time * 1000:.0f}ms  (환산 {1 / avg_time if avg_time > 0 else 0:.1f} FPS)")

print("\n=== 요약 ===")
for s in summary:
    print(f"{s['backend']:12s}  평균 검출 인원 {s['avg_faces']:5.1f}명   "
          f"평균 {s['avg_time_ms']:6.0f}ms/frame   환산 {s['fps_equiv']:.1f} FPS")

if len(summary) == 2 and summary[0]['avg_faces'] > 0:
    diff = summary[1]['avg_faces'] - summary[0]['avg_faces']
    if abs(diff) < 0.05:
        print(f"\n{summary[1]['backend']}와 {summary[0]['backend']}의 프레임당 평균 검출 인원이 거의 같음")
    else:
        print(f"\n{summary[1]['backend']}가 {summary[0]['backend']}보다 프레임당 평균 {diff:+.1f}명 "
              f"{'더 많이' if diff > 0 else '더 적게'} 검출함")

# CSV 저장 (포트폴리오용 기록)
base = os.path.splitext(os.path.basename(video_path))[0]
csv_path = os.path.join(REPORTS_DIR, f'{base}_backend_compare.csv')
with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=['backend', 'avg_faces', 'total_faces', 'avg_time_ms', 'fps_equiv'])
    w.writeheader()
    w.writerows(summary)
print(f"\nSaved: {csv_path}")
