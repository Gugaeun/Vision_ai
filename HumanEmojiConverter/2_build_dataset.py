import pandas as pd
import os
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

save_dir = './pose_img/person/'
keypoint_path = f'{save_dir}keypoints.csv'
dataset_dir = f'{save_dir}/'
classes_file = f'{save_dir}classes.json'

if not os.path.exists(classes_file):
    print(f"Error: {classes_file} 가 없습니다. 자세 목록을 먼저 작성하세요.")
    exit()

with open(classes_file, encoding='utf-8') as fp:
    pose_classes = json.load(fp)

# 이미지 이름 -> 자세 이름 조회표 생성 (실제 폴더 위치가 최종 기준이 됨.
# 잘못 찍힌 사진은 폴더만 옮기고 이 스크립트를 다시 돌리면 라벨이 바로잡힘)
image_to_label = {}
empty_poses = []
for pose_name in pose_classes:
    pose_path = os.path.join(dataset_dir, pose_name)
    if not os.path.isdir(pose_path):
        empty_poses.append(pose_name)
        print(f"  {pose_name}: 폴더 없음")
        continue

    image_names = [f for f in os.listdir(pose_path) if f.lower().endswith('.jpg')]
    if not image_names:
        empty_poses.append(pose_name)
    for image_name in image_names:
        image_to_label[image_name] = pose_name
    print(f"  {pose_name}: {len(image_names)}장")

if empty_poses:
    print(f"Warning: 이미지가 없는 자세가 있습니다 - {empty_poses}")

df = pd.read_csv(keypoint_path)

# 폴더 위치 기준으로 라벨을 다시 씀 (1_collect_pose.py가 적어둔 label 컬럼은 무시)
df['label'] = df['image_name'].map(image_to_label)
df = df.dropna(subset=['label'])

df.to_csv(f'{dataset_dir}dataset.csv', index=False)
print(f"Saved: {dataset_dir}dataset.csv ({len(df)}행, 자세 {df['label'].nunique()}종)")
print(df['label'].value_counts().to_string())
