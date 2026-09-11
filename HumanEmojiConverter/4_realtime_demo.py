# pandas는 ultralytics보다 먼저 임포트해야 함 (ultralytics가 먼저 올라오면 pyarrow 임포트가 WinError 6714로 실패함)
import pandas as pd
from ultralytics import YOLO
import cv2
import numpy as np
import xgboost as xgb
import os
import json
from PIL import Image, ImageDraw, ImageFont

from features import compute_derived_features

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
Models_dir = os.path.abspath(os.path.join(script_dir, "../00_Models"))
model_file = os.path.join(Models_dir, "yolov8n-pose.pt")

save_dir = './pose_img/person/'
weight_file = f'{save_dir}model_weights.xgb'
classes_file = f'{save_dir}classes.json'
feature_info_file = f'{save_dir}feature_columns.json'

model_yolo = YOLO(model_file)

action_model = xgb.XGBClassifier()
action_model.load_model(weight_file)

if not os.path.exists(classes_file):
    print(f"Error: {classes_file} 가 없습니다.")
    exit()
with open(classes_file, encoding='utf-8') as fp:
    pose_classes = json.load(fp)

if not os.path.exists(feature_info_file):
    print(f"Error: {feature_info_file} 가 없습니다. 3_train.py를 먼저 실행하세요.")
    exit()
with open(feature_info_file, encoding='utf-8') as fp:
    feature_cols = json.load(fp)

print(f"자세 {len(pose_classes)}종: {pose_classes}")

# 자세 -> (한글 라벨, 얼굴 색상 BGR)
MOOD_INFO = {
    'thinking': ('생각 중', (255, 200, 0)),
    'excited': ('신남!', (0, 165, 255)),
    'tired': ('지침...', (150, 150, 150)),
}

FONT_PATH = "C:\\Windows\\Fonts\\malgun.ttf"
font = ImageFont.truetype(FONT_PATH, 28)


def draw_emoji_face(frame, center, radius, mood, color):
    """cv2 도형만으로 간단한 이모지 얼굴을 그림 (폰트/유니코드 이모지 의존 없이 항상 동일하게 렌더링됨)."""
    x, y = center
    outline = (0, 0, 0)
    cv2.circle(frame, (x, y), radius, color, -1)
    cv2.circle(frame, (x, y), radius, outline, 2)

    eye_dx, eye_dy = radius // 3, radius // 4
    eye_r = max(3, radius // 8)

    if mood == 'thinking':
        cv2.circle(frame, (x - eye_dx, y - eye_dy), eye_r, outline, -1)
        cv2.circle(frame, (x + eye_dx, y - eye_dy), eye_r, outline, -1)
        cv2.line(frame, (x - radius // 3, y + radius // 2),
                 (x + radius // 4, y + radius // 2 - 6), outline, 3)
    elif mood == 'excited':
        cv2.circle(frame, (x - eye_dx, y - eye_dy), eye_r + 2, outline, -1)
        cv2.circle(frame, (x + eye_dx, y - eye_dy), eye_r + 2, outline, -1)
        cv2.circle(frame, (x - eye_dx + 2, y - eye_dy - 2), 2, (255, 255, 255), -1)
        cv2.circle(frame, (x + eye_dx + 2, y - eye_dy - 2), 2, (255, 255, 255), -1)
        cv2.ellipse(frame, (x, y + radius // 3), (radius // 2, radius // 3), 0, 0, 180, outline, 3)
    elif mood == 'tired':
        cv2.ellipse(frame, (x - eye_dx, y - eye_dy), (eye_r + 2, eye_r), 0, 200, 340, outline, 2)
        cv2.ellipse(frame, (x + eye_dx, y - eye_dy), (eye_r + 2, eye_r), 0, 200, 340, outline, 2)
        cv2.ellipse(frame, (x, y + radius // 2), (radius // 4, radius // 5), 0, 0, 360, outline, 2)


def draw_korean_text(frame, text, pos, color_rgb):
    img_pillow = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pillow)
    draw.text(pos, text, color_rgb, font=font)
    return np.array(img_pillow)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: 카메라를 열 수 없습니다.")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model_yolo(frame, verbose=False)
    annotated_frame = results[0].plot(boxes=False)

    for r in results:
        bound_box = r.boxes.xyxy
        conf = r.boxes.conf.tolist()
        keypoints = r.keypoints.xyn.tolist()

        for index, box in enumerate(bound_box):
            if conf[index] > 0.75:
                x1, y1, x2, y2 = box.tolist()

                data = {}
                for j in range(len(keypoints[index])):
                    data[f'x{j}'] = keypoints[index][j][0]
                    data[f'y{j}'] = keypoints[index][j][1]
                data.update(compute_derived_features(data))

                row = pd.DataFrame([data])[feature_cols]
                pred = int(action_model.predict(row)[0])
                pose_name = pose_classes[pred]
                label_ko, color = MOOD_INFO.get(pose_name, (pose_name, (255, 255, 255)))

                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # 사람 머리 위에 미니 이모지 얼굴을 띄움
                face_cx = int((x1 + x2) / 2)
                face_cy = max(30, int(y1) - 45)
                draw_emoji_face(annotated_frame, (face_cx, face_cy), 30, pose_name, color)

                annotated_frame = draw_korean_text(
                    annotated_frame, label_ko, (int(x1), max(0, int(y1) - 85)),
                    (color[2], color[1], color[0]))  # BGR -> RGB로 바꿔서 PIL에 전달

    cv2.imshow("Human Emoji Converter", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
