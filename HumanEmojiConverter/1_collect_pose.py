# pandas는 ultralytics보다 먼저 임포트해야 함 (ultralytics가 먼저 올라오면 pyarrow 임포트가 WinError 6714로 실패함)
import pandas as pd
import cv2
from ultralytics import YOLO
import os
import sys
import time
import json

#=================================================================
# 초기값 설정
#=================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
Models_dir = os.path.abspath(os.path.join(script_dir, "../00_Models"))
model_file = os.path.join(Models_dir, "yolov8n-pose.pt")

save_dir = './pose_img/person/'
keypoint_path = f'{save_dir}keypoints.csv'
classes_file = f'{save_dir}classes.json'

if not os.path.exists(classes_file):
    print(f"Error: {classes_file} 가 없습니다. 자세 목록을 먼저 작성하세요.")
    exit()

with open(classes_file, encoding='utf-8') as fp:
    pose_classes = json.load(fp)

# 이번에 수집할 클래스를 커맨드라인 인자로 받음 (예: python 1_collect_pose.py thinking)
if len(sys.argv) < 2:
    print(f"사용법: python 1_collect_pose.py <클래스명>")
    print(f"사용 가능한 클래스: {pose_classes}")
    exit()

target_class = sys.argv[1]
if target_class not in pose_classes:
    print(f"Error: '{target_class}'는 classes.json에 없는 클래스입니다. 사용 가능: {pose_classes}")
    exit()

class_dir = os.path.join(save_dir, target_class)
os.makedirs(class_dir, exist_ok=True)

# 이미 저장된 이미지 수 이후부터 번호를 이어감 (여러 번 나눠 찍어도 덮어쓰지 않음)
existing = [f for f in os.listdir(class_dir) if f.lower().endswith('.jpg')]
start_index = len(existing)
print(f"'{target_class}' 클래스: 기존 {start_index}장에 이어서 수집합니다.")

# YOLOv8 포즈 추정 모델 불러오기
model = YOLO(model_file)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: 카메라를 열 수 없습니다.")
    exit()

# 수집 간격(초)과 수집 목표 장수 설정 (필요에 따라 변경 가능)
interval_sec = 0.3
frame_total = 150
last_t = 0
i = 0
a = start_index

# 이번 실행에서 새로 만든 데이터만 담는 리스트 (기존 keypoints.csv에는 append함)
new_data = []

while cap.isOpened():
    flag, frame = cap.read()
    if not flag:
        break
    if i >= frame_total:
        break

    now = time.time()
    collect = (now - last_t) >= interval_sec

    results = model(frame, verbose=False)
    view = results[0].plot()

    if collect:
        last_t = now
        for r in results:
            bound_box = r.boxes.xyxy
            conf = r.boxes.conf.tolist()
            keypoints = r.keypoints.xyn.tolist()

            for index, box in enumerate(bound_box):
                if conf[index] > 0.75:
                    x1, y1, x2, y2 = box.tolist()
                    pict = frame[int(y1):int(y2), int(x1):int(x2)]

                    image_name = f'{target_class}_{a}.jpg'
                    output_path = os.path.join(class_dir, image_name)

                    data = {'image_name': image_name, 'label': target_class}
                    for j in range(len(keypoints[index])):
                        data[f'x{j}'] = keypoints[index][j][0]
                        data[f'y{j}'] = keypoints[index][j][1]

                    new_data.append(data)
                    cv2.imwrite(output_path, pict)
                    a += 1

        i += 1

    cv2.putText(view, f"[{target_class}] collect {i}/{frame_total}  saved {a - start_index}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("Lifting Pose Collect", view)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print(f"수집 완료: {i}프레임 처리, {a - start_index}장 저장 (누적 {a}장)")
cap.release()
cv2.destroyAllWindows()

# keypoints.csv에 append (없으면 새로 생성)
new_df = pd.DataFrame(new_data)
if os.path.exists(keypoint_path):
    new_df.to_csv(keypoint_path, mode='a', header=False, index=False)
else:
    new_df.to_csv(keypoint_path, mode='w', header=True, index=False)
print(f"Saved: {keypoint_path} (+{len(new_df)}행)")
