import cv2
import numpy as np


def segment_furrow(bgr_image):
    """
    ExG(Excess Green Index) 기반 규칙형 고랑(작물 재배 영역) 분할.
    학습 데이터 없이, "초록색이 많이 섞인 영역"을 식생 영역으로 보고
    그 중 화면 하단부에 넓게 퍼진 덩어리를 고랑으로 간주한다.

    반환: mask (uint8, 0/255), shape = (H, W) — 255인 곳이 고랑(작물 재배 영역)
    """
    img = bgr_image.astype(np.float32)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # ExG = 2*G - R - B : 식생(초록)이 강조되고 흙/그림자는 약해짐
    exg = 2 * g - r - b
    exg = np.clip(exg, 0, None)
    exg_u8 = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Otsu로 식생/배경 자동 분리
    _, veg_mask = cv2.threshold(exg_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 작은 잡음 제거 + 구멍 메우기 (고랑은 연속된 큰 영역이어야 함)
    kernel = np.ones((9, 9), np.uint8)
    veg_mask = cv2.morphologyEx(veg_mask, cv2.MORPH_OPEN, kernel)
    veg_mask = cv2.morphologyEx(veg_mask, cv2.MORPH_CLOSE, kernel)

    # 가장 큰 연결 성분들(식생 덩어리)만 고랑으로 채택 — 작은 잡초 점 하나가
    # 고랑처럼 인식되는 것을 막기 위함 (면적 하위 5%는 버림)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(veg_mask, connectivity=8)
    furrow_mask = np.zeros_like(veg_mask)
    if n_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        area_thresh = max(200, np.percentile(areas, 50))
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] >= area_thresh:
                furrow_mask[labels == i] = 255

    # 실제 밭 사진은 포기 사이사이 흙이 드러나 있어서 위 결과가 듬성듬성 끊긴
    # "구멍 뚫린" 형태가 되기 쉽다. 고랑은 원래 연속된 재배 줄이므로,
    # 팽창(dilate)으로 개별 포기 사이 간격을 메워 하나의 이어진 영역으로 만든다.
    dilate_kernel = np.ones((25, 25), np.uint8)
    furrow_mask = cv2.dilate(furrow_mask, dilate_kernel)

    return furrow_mask
