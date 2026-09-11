import numpy as np
import pandas as pd

# COCO 17 keypoint indices (YOLOv8-pose 기준)
NUM_KEYPOINTS = 17
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10

# 어깨 중심을 원점(0,0), 어깨 너비를 1로 맞춘 정규화 좌표 (nx0,ny0 ... nx16,ny16)
NORMALIZED_KEYPOINT_NAMES = []
for _j in range(NUM_KEYPOINTS):
    NORMALIZED_KEYPOINT_NAMES.append(f'nx{_j}')
    NORMALIZED_KEYPOINT_NAMES.append(f'ny{_j}')

GESTURE_FEATURE_NAMES = ['hand_face_dist_min', 'hand_face_dist_max', 'hands_raised']
DERIVED_FEATURE_NAMES = NORMALIZED_KEYPOINT_NAMES + GESTURE_FEATURE_NAMES


def compute_derived_features(row):
    """
    row: x0..x16, y0..y16 (정규화된 keypoint 좌표)을 담은 dict 또는 pandas Series

    원본 x0..y16은 "화면 전체 기준" 좌표라, 카메라와의 거리가 달라지면(줌인/줌아웃,
    의자를 앞뒤로 움직임 등) 같은 동작이어도 절대 좌표값이 달라진다. 그래서 촬영 거리가
    다른 세션에서 모은 데이터를 섞으면 모델이 헷갈릴 수 있다.

    여기서는 어깨 중심을 원점으로, 어깨 너비를 1로 맞춰 모든 keypoint를 다시 계산한다
    (= "몸 크기 대비 비율" 좌표). 카메라가 가깝든 멀든 이 비율은 거의 그대로 유지되므로
    더 안정적이다. 손-얼굴 거리, 손 들어올림 정도도 이 정규화된 좌표로 다시 계산한다.
    """
    lsx, lsy = row[f'x{L_SHOULDER}'], row[f'y{L_SHOULDER}']
    rsx, rsy = row[f'x{R_SHOULDER}'], row[f'y{R_SHOULDER}']
    cx, cy = (lsx + rsx) / 2, (lsy + rsy) / 2
    scale = max(1e-6, float(np.hypot(lsx - rsx, lsy - rsy)))  # 어깨 너비 (0이 되는 것 방지)

    normed = {}
    for j in range(NUM_KEYPOINTS):
        normed[f'nx{j}'] = float((row[f'x{j}'] - cx) / scale)
        normed[f'ny{j}'] = float((row[f'y{j}'] - cy) / scale)

    def is_missing(idx):
        # YOLO-pose는 가려지거나 화면 밖으로 나간 keypoint를 (0,0)으로 반환하는 경우가 있다.
        # 사람이 이미지 맨 왼쪽 위 모서리(0,0)에 실제로 있을 리는 없으므로, 이걸 "미검출"로 간주한다.
        # (이 체크를 안 하면, 손이 프레임 밖으로 나갔을 때 정규화 좌표가 "어깨보다 훨씬 위"로
        #  계산돼서 가만히 있어도 excited로 오판하는 버그가 생긴다.)
        return abs(row[f'x{idx}']) < 1e-6 and abs(row[f'y{idx}']) < 1e-6

    nx, ny = normed[f'nx{NOSE}'], normed[f'ny{NOSE}']
    lwx, lwy = normed[f'nx{L_WRIST}'], normed[f'ny{L_WRIST}']
    rwx, rwy = normed[f'nx{R_WRIST}'], normed[f'ny{R_WRIST}']

    FAR = 100.0          # 손-얼굴 거리 미검출 시 "확실히 멀다"로 취급
    NOT_RAISED = -100.0  # 손 들어올림 미검출 시 "확실히 안 들었다"로 취급

    dist_l = FAR if is_missing(L_WRIST) else float(np.hypot(lwx - nx, lwy - ny))
    dist_r = FAR if is_missing(R_WRIST) else float(np.hypot(rwx - nx, rwy - ny))

    # 한 손만 얼굴 가까이: thinking(턱 괴기)일 때 작아짐 (반대쪽 손은 멀어도 상관없음)
    hand_face_dist_min = min(dist_l, dist_r)
    # 두 손 다 얼굴 가까이: tired(양손으로 얼굴 가리기)일 때 이 값도 작아짐.
    # thinking은 한쪽 손만 가까이 가므로 이 값은 여전히 큼 -> 두 클래스를 구분하는 핵심 특징.
    hand_face_dist_max = max(dist_l, dist_r)

    # 손 들어올림 정도: 어깨 중심(정규화 후 y=0) 대비 손목이 얼마나 위에 있는지.
    # 양손 들기(excited)일 때 커짐. 두 손 다 올라가야 커지도록 min 사용.
    raised_l = NOT_RAISED if is_missing(L_WRIST) else -lwy
    raised_r = NOT_RAISED if is_missing(R_WRIST) else -rwy
    hands_raised = float(min(raised_l, raised_r))

    result = dict(normed)
    result['hand_face_dist_min'] = hand_face_dist_min
    result['hand_face_dist_max'] = hand_face_dist_max
    result['hands_raised'] = hands_raised
    return result


def add_derived_features(df):
    """DataFrame에 파생 특징 컬럼을 추가해서 반환."""
    derived = df.apply(compute_derived_features, axis=1, result_type='expand')
    return pd.concat([df, derived], axis=1)
