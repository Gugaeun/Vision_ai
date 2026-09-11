import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import json

from features import add_derived_features, DERIVED_FEATURE_NAMES

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

save_dir = './pose_img/person/'
dataset_file = f'{save_dir}dataset.csv'
weight_file = f'{save_dir}model_weights.xgb'
classes_file = f'{save_dir}classes.json'
feature_info_file = f'{save_dir}feature_columns.json'

df = pd.read_csv(dataset_file)

with open(classes_file, encoding='utf-8') as fp:
    pose_classes = json.load(fp)
label_to_id = {name: i for i, name in enumerate(pose_classes)}

unknown = sorted(set(df['label']) - set(pose_classes))
if unknown:
    print(f"Warning: classes.json에 없는 라벨을 제외합니다 - {unknown}")
    df = df[df['label'].isin(pose_classes)]

empty_poses = [name for name in pose_classes if name not in set(df['label'])]
if empty_poses:
    print(f"Error: 학습 데이터가 없는 자세가 있습니다 - {empty_poses}")
    exit()

print(f"자세 {len(pose_classes)}종: {pose_classes}")
print(df['label'].value_counts().to_string())

# 파생 특징(몸통 기울기, 좌우 비틀림) 추가
df = add_derived_features(df)

raw_cols = [c for c in df.columns if c not in ('image_name', 'label') and c not in DERIVED_FEATURE_NAMES]
y = df['label'].map(label_to_id)


def train_and_eval(feature_cols, tag):
    X = df[feature_cols]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    objective, eval_metric = ('binary:logistic', 'logloss') if len(pose_classes) == 2 \
        else ('multi:softmax', 'mlogloss')
    model = xgb.XGBClassifier(objective=objective, eval_metric=eval_metric)
    model.fit(X_train, y_train, verbose=False)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[{tag}] feature 수={len(feature_cols)}  accuracy={acc:.4f}")
    return model, acc


# 비교 1: 원본 keypoint 좌표만 사용
_, acc_raw = train_and_eval(raw_cols, "원본 keypoint만")

# 비교 2: 원본 + 파생 특징(몸통 기울기, 좌우 비틀림)
final_cols = raw_cols + DERIVED_FEATURE_NAMES
final_model, acc_derived = train_and_eval(final_cols, "원본 + 파생 특징")

print("=" * 50)
print(f"파생 특징 추가 전 정확도: {acc_raw:.4f}")
print(f"파생 특징 추가 후 정확도: {acc_derived:.4f}")
print(f"변화: {acc_derived - acc_raw:+.4f}")
print("=" * 50)

# 최종 모델(파생 특징 포함)로 저장. 실시간 데모에서 같은 컬럼 순서로 특징을 만들어야 하므로 컬럼 목록도 저장.
final_model.save_model(weight_file)
with open(feature_info_file, 'w', encoding='utf-8') as fp:
    json.dump(final_cols, fp, ensure_ascii=False, indent=2)

print(f"Saved: {weight_file}")
print(f"Saved: {feature_info_file}")
