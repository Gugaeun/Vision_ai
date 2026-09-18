import cv2
import numpy as np

# 확장 시 이 경로에 학습된 YOLOv8 (crop/weed 2class) 가중치를 두면
# detect_plants()가 자동으로 규칙 기반 대신 이 모델을 사용하도록 바뀐다.
YOLO_WEIGHTS_PATH = "models/yolov8_crop_weed.pt"

_yolo_model = None
_yolo_load_attempted = False


def _try_load_yolo():
    global _yolo_model, _yolo_load_attempted
    if _yolo_load_attempted:
        return _yolo_model
    _yolo_load_attempted = True
    import os
    if not os.path.exists(YOLO_WEIGHTS_PATH):
        return None
    try:
        from ultralytics import YOLO
        _yolo_model = YOLO(YOLO_WEIGHTS_PATH)
    except Exception as e:
        print(f"YOLO 가중치를 불러오지 못해 규칙 기반 탐지로 대체합니다: {e}")
        _yolo_model = None
    return _yolo_model


def detect_plants_yolo(bgr_image, conf_threshold=0.3):
    """학습된 YOLOv8 (crop/weed) 모델로 개별 식물체 탐지. 없으면 None 반환."""
    model = _try_load_yolo()
    if model is None:
        return None

    results = model.predict(bgr_image, conf=conf_threshold, verbose=False)
    r = results[0]
    names = r.names

    plants = []
    for box in r.boxes:
        cls_id = int(box.cls[0])
        label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        plants.append({"label": label, "confidence": round(conf, 3), "box": [x1, y1, x2, y2]})
    return plants


def detect_plants_rule_based(bgr_image):
    """
    MVP 규칙 기반 개별 식물체 탐지 + crop/weed 분류 (학습 데이터 불필요).

    1) ExG로 식생 픽셀을 찾고, 연결 성분마다 하나의 "개체"로 취급.
    2) 화면 전체 식생 덩어리들의 중앙값 면적을 기준으로,
       면적이 크고 둥글게 뭉친(볼록도가 높은) 개체는 보통 관리된 작물,
       작고 삐죽삐죽한(볼록도가 낮은) 개체는 잡초로 보는 휴리스틱을 적용.
       (실제 서비스에서는 이 부분을 YOLOv8 crop/weed 분류 모델로 교체 — YOLO_WEIGHTS_PATH 참고)
    """
    img = bgr_image.astype(np.float32)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    exg = np.clip(2 * g - r - b, 0, None)
    exg_u8 = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, veg_mask = cv2.threshold(exg_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    veg_mask = cv2.morphologyEx(veg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    img_h, img_w = bgr_image.shape[:2]
    max_plant_area = 0.08 * img_h * img_w  # 이보다 크면 "포기 하나"가 아니라 여러 포기가 붙은 덩어리로 간주

    contours, _ = cv2.findContours(veg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 60:  # 노이즈 점 제거
            continue
        if area > max_plant_area:
            # 캐노피가 겹쳐서 여러 포기가 뭉친 덩어리 — 개별 개체로 볼 수 없으므로 제외.
            # (이걸 포함하면 화면 절반을 덮는 박스가 잡초 하나로 오판되는 문제가 생긴다.)
            continue
        hull = cv2.convexHull(c)
        hull_area = max(cv2.contourArea(hull), 1e-6)
        solidity = area / hull_area
        x, y, w, h = cv2.boundingRect(c)
        candidates.append({"box": [x, y, x + w, y + h], "area": area, "solidity": solidity})

    if not candidates:
        return []

    areas = np.array([c["area"] for c in candidates])
    median_area = float(np.median(areas))

    plants = []
    for c in candidates:
        is_crop_like = (c["area"] >= median_area * 0.8) and (c["solidity"] >= 0.55)
        label = "crop" if is_crop_like else "weed"
        # 규칙 기반 점수(확신도 아님) — 면적비와 볼록도를 0~1로 합성
        score = min(1.0, 0.5 * (c["area"] / (median_area * 2 + 1e-6)) + 0.5 * c["solidity"])
        plants.append({"label": label, "confidence": round(float(score), 3), "box": c["box"]})
    return plants


def detect_plants(bgr_image, conf_threshold=0.3):
    """YOLOv8 crop/weed 가중치가 있으면 그걸 쓰고, 없으면 규칙 기반으로 대체."""
    yolo_result = detect_plants_yolo(bgr_image, conf_threshold)
    if yolo_result is not None:
        return yolo_result, "yolo"
    return detect_plants_rule_based(bgr_image), "rule_based"
