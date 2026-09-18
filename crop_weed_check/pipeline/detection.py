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


DENSITY_THRESHOLD = 0.30  # 이 이상이면 "빽빽하게 심긴 줄 안"으로 보고 crop 판정


def detect_plants_rule_based(bgr_image):
    """
    MVP 규칙 기반 개별 식물체 탐지 + crop/weed 분류 (학습 데이터 불필요).

    1) ExG로 식생 픽셀을 찾고, 연결 성분마다 하나의 "개체"로 취급.
    2) crop/weed 판정은 "주변 밀도"로 한다 — 실제로 심어진 작물은 촘촘하게 줄지어
       자라서 주변에도 식생 픽셀이 빽빽하고(캐노피가 이어짐), 잡초는 이랑 사이
       빈 흙에 듬성듬성 홀로 자라서 주변이 휑하다는 관찰에 기반한다.
       (처음엔 "면적·볼록도"로 판정했는데, 실제 사진에서는 작물 잎 하나하나의
        모양이 제각각이라 그 기준만으로는 오판이 너무 많았다 — 아래
        "겪었던 문제" README 참고)
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

    # 화면을 절반 가까이 덮는 큰 덩어리 = 여러 포기가 이어져 자란 "작물 줄(캐노피)" 그 자체.
    # 이 덩어리에 직접 맞닿아 있는 개체는 사실상 그 줄의 일부(줄 맨 끝 포기 등)인데,
    # 밀도 계산용 사각 윈도우는 한쪽 방향(줄 바깥쪽)에 이웃이 없어서 밀도가 낮게 나오고
    # weed로 오판되는 경우가 있었다. 그래서 "큰 캐노피에 맞닿아 있는가"를 밀도보다 먼저 본다.
    n_labels, labels_img, stats, _ = cv2.connectedComponentsWithStats(veg_mask, connectivity=8)
    canopy_mask = np.zeros_like(veg_mask)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_plant_area:
            canopy_mask[labels_img == i] = 255
    canopy_mask = cv2.dilate(canopy_mask, np.ones((15, 15), np.uint8))

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
        x, y, w, h = cv2.boundingRect(c)
        candidates.append({"box": [x, y, x + w, y + h], "area": area})

    if not candidates:
        return []

    plants = []
    for c in candidates:
        x1, y1, x2, y2 = c["box"]

        touches_canopy = bool(np.count_nonzero(canopy_mask[y1:y2, x1:x2]))
        if touches_canopy:
            # 큰 작물 줄 캐노피에 직접 맞닿은 개체는 밀도 계산 없이 바로 crop으로 판정.
            is_crop_like = True
            density = 1.0
        else:
            bw, bh = x2 - x1, y2 - y1
            # 개체 크기의 1.5배만큼 여유를 두고 주변 영역을 잘라 식생 밀도를 잰다.
            pad_x, pad_y = int(bw * 1.5) + 10, int(bh * 1.5) + 10
            wx1, wy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            wx2, wy2 = min(img_w, x2 + pad_x), min(img_h, y2 + pad_y)
            window = veg_mask[wy1:wy2, wx1:wx2]
            density = float(np.count_nonzero(window)) / window.size if window.size else 0.0
            is_crop_like = density >= DENSITY_THRESHOLD

        label = "crop" if is_crop_like else "weed"
        score = min(1.0, density / (DENSITY_THRESHOLD * 2))
        plants.append({"label": label, "confidence": round(score, 3), "box": c["box"]})
    return plants


def detect_plants(bgr_image, conf_threshold=0.3):
    """YOLOv8 crop/weed 가중치가 있으면 그걸 쓰고, 없으면 규칙 기반으로 대체."""
    yolo_result = detect_plants_yolo(bgr_image, conf_threshold)
    if yolo_result is not None:
        return yolo_result, "yolo"
    return detect_plants_rule_based(bgr_image), "rule_based"
