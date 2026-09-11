import sys

# deepface가 모델을 처음 내려받을 때 이모지가 섞인 로그를 출력하는데, Windows 콘솔 기본
# 인코딩(cp949)로는 이걸 출력하지 못해 다운로드 자체는 성공했는데도 프로그램이 죽는 경우가 있음.
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

ANALYZE_EVERY_SEC = 1.0  # 이 간격으로 한 프레임씩만 분석 (군중 영상은 얼굴이 많아 매 프레임 분석하면 매우 느림)
BACKEND = 'retinaface'   # 작고 겹친 얼굴이 많은 군중 영상은 retinaface가 opencv보다 훨씬 잘 찾음
MAX_DISPLAY_WIDTH = 960  # 재생 창 최대 가로 크기(px). 원본이 더 크면 화면 표시만 이 크기로 줄임

AGE_BINS = [0, 10, 20, 30, 40, 50, 60, 200]
AGE_LABELS = ['0-9', '10s', '20s', '30s', '40s', '50s', '60+']


def age_bucket(age):
    for i in range(len(AGE_BINS) - 1):
        if AGE_BINS[i] <= age < AGE_BINS[i + 1]:
            return AGE_LABELS[i]
    return AGE_LABELS[-1]


def find_video():
    files = sorted(glob.glob(os.path.join(VIDEOS_DIR, '*.mp4')) +
                    glob.glob(os.path.join(VIDEOS_DIR, '*.avi')) +
                    glob.glob(os.path.join(VIDEOS_DIR, '*.mov')))
    return files[0] if files else None


if len(sys.argv) >= 2:
    video_path = sys.argv[1]
else:
    video_path = find_video()
    if not video_path:
        print(f"Error: {VIDEOS_DIR}/ 에 분석할 영상이 없습니다.")
        print("공개된(저작권 문제 없는) 군중/관객 영상을 mp4로 넣어주세요. (예: Pexels, Pixabay 등 무료 영상 사이트)")
        exit()
    print(f"영상 지정 없음 - {VIDEOS_DIR}/ 안의 첫 영상 사용: {video_path}")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: 영상을 열 수 없습니다 - {video_path}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) or 30
analyze_every_n = max(1, round(fps * ANALYZE_EVERY_SEC))

log = []  # [{'t':, 'age':, 'age_bucket':, 'gender':, 'dominant_emotion':}, ...] (한 얼굴당 한 행)
frame_idx = 0
last_faces = []  # 최근 분석 프레임에서 찾은 얼굴 목록 (화면 표시용)

print("영상 재생 + 분석을 시작합니다 ('q'로 중단 가능).")
proc_times = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    video_t = frame_idx / fps

    if frame_idx % analyze_every_n == 0:
        t0 = time.time()
        try:
            results = DeepFace.analyze(frame, actions=['age', 'gender', 'emotion'],
                                        detector_backend=BACKEND,
                                        enforce_detection=False, silent=True)
            results = [r for r in results if r.get('face_confidence', 1) > 0]
        except Exception:
            results = []
        proc_times.append(time.time() - t0)

        last_faces = results
        for r in results:
            log.append({
                't': round(video_t, 2),
                'age': r['age'],
                'age_bucket': age_bucket(r['age']),
                'gender': r['dominant_gender'],
                'dominant_emotion': r['dominant_emotion'],
            })

        # 분석이 끝난 프레임만 화면에 표시함. 매 프레임을 다 보여주면 분석 중(수백ms~수초)에
        # 화면이 멈춰서 "정상 재생 -> 뚝 멈춤"이 반복되는 것처럼 보임 -> 분석 프레임만 보여주는
        # 슬라이드쇼 방식으로 바꿔서, 느리지만 끊김 없이 갱신되는 것처럼 보이게 함.
        for r in last_faces:
            region = r['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            label = f"{r['dominant_gender'][0]},{r['age']},{r['dominant_emotion']}"
            cv2.putText(frame, label, (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cur_fps = 1 / proc_times[-1] if proc_times and proc_times[-1] > 0 else 0
        cv2.putText(frame, f"t={video_t:.1f}s  faces={len(last_faces)}  analyze FPS={cur_fps:.1f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 화면 표시용으로만 축소 (분석은 원본 해상도로 계속함) - 고해상도 영상은 창이 화면보다 커짐
        display_frame = frame
        if frame.shape[1] > MAX_DISPLAY_WIDTH:
            scale = MAX_DISPLAY_WIDTH / frame.shape[1]
            display_frame = cv2.resize(frame, (MAX_DISPLAY_WIDTH, int(frame.shape[0] * scale)))
        cv2.imshow("Crowd Demographics", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
print(f"\n분석 완료: 총 {len(log)}건의 얼굴 감지 (분석한 프레임 {len(proc_times)}장)")

if not log:
    print("얼굴이 하나도 감지되지 않았습니다. 영상이나 임계값을 확인하세요.")
    exit()

# ---- 통계 집계 ----
gender_counts = {}
emotion_counts = {}
age_counts = {}
for e in log:
    gender_counts[e['gender']] = gender_counts.get(e['gender'], 0) + 1
    emotion_counts[e['dominant_emotion']] = emotion_counts.get(e['dominant_emotion'], 0) + 1
    age_counts[e['age_bucket']] = age_counts.get(e['age_bucket'], 0) + 1

n = len(log)
print("\n=== 성별 분포 ===")
for k, v in sorted(gender_counts.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}건 ({v / n * 100:.1f}%)")

print("\n=== 연령대 분포 ===")
for label in AGE_LABELS:
    v = age_counts.get(label, 0)
    if v:
        print(f"  {label}: {v}건 ({v / n * 100:.1f}%)")

print("\n=== 감정 분포 ===")
for k, v in sorted(emotion_counts.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}건 ({v / n * 100:.1f}%)")

# ---- CSV 저장 ----
base = os.path.splitext(os.path.basename(video_path))[0]
csv_path = os.path.join(REPORTS_DIR, f'{base}_faces.csv')
with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=['t', 'age', 'age_bucket', 'gender', 'dominant_emotion'])
    w.writeheader()
    w.writerows(log)
print(f"\nSaved: {csv_path}")

# ---- 요약 그래프 저장 ----
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].bar(gender_counts.keys(), gender_counts.values(), color=['#4C72B0', '#DD8452'])
    axes[0].set_title('Gender Distribution')

    axes[1].bar([l for l in AGE_LABELS if l in age_counts],
                [age_counts[l] for l in AGE_LABELS if l in age_counts], color='#55A868')
    axes[1].set_title('Age Distribution')
    axes[1].tick_params(axis='x', rotation=45)

    emo_sorted = sorted(emotion_counts.items(), key=lambda x: -x[1])
    axes[2].bar([e[0] for e in emo_sorted], [e[1] for e in emo_sorted], color='#C44E52')
    axes[2].set_title('Emotion Distribution')
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plot_path = os.path.join(REPORTS_DIR, f'{base}_summary.png')
    plt.savefig(plot_path)
    print(f"Saved: {plot_path}")
except ImportError:
    print("matplotlib이 없어 그래프는 건너뜁니다 (pip install matplotlib)")
