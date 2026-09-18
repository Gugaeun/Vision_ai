import numpy as np


def judge_treatment_targets(furrow_mask, plants):
    """
    필수 요건 2: 세그멘테이션 결과(고랑 마스크) ∩ 탐지 결과(weed 박스) → 방제 대상 판단.

    규칙: "고랑(작물 재배 영역) 안인데 weed로 분류된 개체"만 방제 대상으로 표시한다.
    - 고랑 밖의 weed는 애초에 밭 관리 대상이 아니므로 무시.
    - 고랑 안의 crop은 당연히 무시.

    furrow_mask: (H, W) uint8, 255=고랑
    plants: [{"label": "crop"|"weed", "confidence": float, "box": [x1,y1,x2,y2]}, ...]

    반환: plants와 같은 리스트지만 각 항목에 "in_furrow"(bool), "treatment_target"(bool) 추가
    """
    h, w = furrow_mask.shape[:2]
    judged = []
    for p in plants:
        x1, y1, x2, y2 = p["box"]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))

        in_furrow = False
        if x2 > x1 and y2 > y1:
            # 박스 중심점이 고랑 마스크 안에 있는지로 판정한다.
            # (면적 절반 이상 겹침 기준을 써봤더니, 실제 밭 사진은 포기 사이 흙 때문에
            #  고랑 마스크에 구멍이 많이 뚫려 있어서 박스가 명백히 고랑 안에 있어도
            #  절반 기준을 못 채우는 경우가 많았다 — 중심점 기준이 훨씬 안정적이었다.)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            in_furrow = bool(furrow_mask[cy, cx] > 0)

        is_weed = p["label"] == "weed"
        treatment_target = in_furrow and is_weed

        judged.append({**p, "in_furrow": bool(in_furrow), "treatment_target": bool(treatment_target)})
    return judged


def summarize(judged_plants):
    n_targets = sum(1 for p in judged_plants if p["treatment_target"])
    n_crop = sum(1 for p in judged_plants if p["label"] == "crop")
    n_weed = sum(1 for p in judged_plants if p["label"] == "weed")
    return {
        "treatment_targets": n_targets,
        "total_crop": n_crop,
        "total_weed": n_weed,
        "total_plants": len(judged_plants),
    }
