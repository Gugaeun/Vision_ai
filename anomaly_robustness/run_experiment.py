"""
2단계: 조명(밝기)·각도(회전) 강건성 실험.

memory bank는 baseline(변형 없는 원본 학습 이미지)으로 "딱 한 번"만 만들고,
이후 모든 조건에서는 재학습하지 않는다 — "테스트 조건만 바뀌었을 때 성능이
얼마나 깨지는가"를 보는 실험이라, 학습 쪽을 고정해야 비교가 공정해진다.

실행:
    python run_experiment.py
"""
import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# 한글 폰트 지정 (안 하면 그래프의 한글 라벨이 네모(□)로 깨진다)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

script_dir = Path(__file__).resolve().parent
base_code_dir = script_dir.parent / "17_Anomoly_Detection"
sys.path.insert(0, str(base_code_dir))
from patchcore_anomaly_detection import build_model, Dataset_dir  # noqa: E402

import os
# patchcore_anomaly_detection.py는 import되는 순간 자기 자신의 위치(17_Anomoly_Detection/)로
# os.chdir 해버린다. 그대로 두면 체크포인트가 원본 폴더 밑에 생겨서 우리가 새로 만든
# 실험 코드와 뒤섞인다. Dataset_dir은 이미 절대경로로 고정된 뒤이므로,
# 여기서 다시 anomaly_robustness/ 로 되돌려서 결과물이 이 폴더 안에만 생기게 한다.
os.chdir(script_dir)

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.metrics import AUROC, PRO

from perturbations import BRIGHTNESS_LEVELS, ROTATION_ANGLES, brightness_transform, rotation_transform

CATEGORY = "hazelnut"
RESULTS_DIR = script_dir / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def make_datamodule(test_augmentations=None):
    # num_workers=0: Windows에서는 워커 프로세스를 띄울 때마다 모듈을 통째로 재import해서
    # 오버헤드가 크다. 이 정도 크기 데이터셋(수백 장)에서는 워커 없이 메인 프로세스에서
    # 바로 읽는 게 더 빠르다.
    return MVTecAD(
        root=Dataset_dir,
        category=CATEGORY,
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,
        test_augmentations=test_augmentations,
    )


def evaluate(model, engine, test_augmentations, condition_name):
    """주어진 변형을 테스트 이미지에 적용한 뒤, 고정된 memory bank로 채점한다."""
    datamodule = make_datamodule(test_augmentations)
    t0 = time.time()
    predictions = engine.predict(model=model, datamodule=datamodule, return_predictions=True)
    elapsed = time.time() - t0

    auroc_metric = AUROC(fields=["pred_score", "gt_label"])
    pro_metric = PRO(fields=["anomaly_map", "gt_mask"])

    n_good, n_good_fp = 0, 0
    for batch in predictions:
        auroc_metric.update(batch)
        pro_metric.update(batch)
        gt_labels = batch.gt_label.tolist()
        pred_labels = batch.pred_label.tolist()
        for gt, pred in zip(gt_labels, pred_labels):
            if gt == 0:  # 정상 이미지
                n_good += 1
                if pred == 1:
                    n_good_fp += 1

    image_auroc = float(auroc_metric.compute())
    pro = float(pro_metric.compute())
    normal_fpr = n_good_fp / n_good if n_good else float("nan")

    print(f"  [{condition_name}] image_AUROC={image_auroc:.4f}  PRO={pro:.4f}  "
          f"정상_오탐율={normal_fpr:.4f} ({n_good_fp}/{n_good})  ({elapsed:.1f}s)")

    return {
        "condition": condition_name,
        "image_AUROC": image_auroc,
        "PRO": pro,
        "normal_FPR": normal_fpr,
        "n_good": n_good,
        "n_good_fp": n_good_fp,
    }, predictions


def save_metrics_csv(rows, path):
    fieldnames = ["condition", "image_AUROC", "PRO", "normal_FPR", "n_good", "n_good_fp"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"저장됨: {path}")


def plot_degradation_curve(x_values, auroc_values, pro_values, xlabel, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(x_values, auroc_values, marker="o", label="Image AUROC")
    ax.plot(x_values, pro_values, marker="s", label="PRO")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"저장됨: {out_path}")


def save_visual_grid(predictions, condition_name, transform, out_path):
    """
    정답/이상맵/예측마스크가 있는 첫 결함(defect) 샘플 하나를 4분할로 저장한다.

    "원본" 패널은 batch.image(모델 입력용으로 정규화된 텐서)가 아니라, 원본 파일을
    다시 읽어 동일한 transform(조명/각도 변형)을 똑같이 적용해서 만든다 — 즉
    "모델이 실제로 받아본 그 변형 이미지"를 그대로 보여주기 위함이다.
    """
    from PIL import Image as PILImage
    from torchvision.io import read_image, ImageReadMode

    for batch in predictions:
        gt_labels = batch.gt_label.tolist()
        for i, gt in enumerate(gt_labels):
            if gt != 1:
                continue
            raw = read_image(batch.image_path[i], mode=ImageReadMode.RGB)
            if transform is not None:
                raw, _ = transform(raw, None)
            image = raw.permute(1, 2, 0).cpu().numpy()
            gt_mask = batch.gt_mask[i].squeeze().cpu().numpy() if batch.gt_mask is not None else None
            anomaly_map = batch.anomaly_map[i].squeeze().cpu().numpy() if batch.anomaly_map is not None else None
            pred_mask = batch.pred_mask[i].squeeze().cpu().numpy() if batch.pred_mask is not None else None

            fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
            axes[0].imshow(image)
            axes[0].set_title("원본")
            if gt_mask is not None:
                axes[1].imshow(gt_mask, cmap="gray")
            axes[1].set_title("정답 마스크")
            if anomaly_map is not None:
                axes[2].imshow(anomaly_map, cmap="jet")
            axes[2].set_title("이상 점수 히트맵")
            if pred_mask is not None:
                axes[3].imshow(pred_mask, cmap="gray")
            axes[3].set_title("예측 마스크")
            for ax in axes:
                ax.axis("off")
            fig.suptitle(f"조건: {condition_name}")
            fig.tight_layout()
            fig.savefig(out_path, dpi=130)
            plt.close(fig)
            print(f"저장됨: {out_path}")
            return
    print(f"경고: {condition_name} 조건에서 결함(defect) 샘플을 찾지 못해 시각화를 건너뜀")


def main():
    print(f"[1/3] baseline 학습 (memory bank 구축, category={CATEGORY}) — 이후 재학습 없음")
    # coreset_ratio를 낮춘 이유는 run_baseline.py 주석 참고 (CPU에서 기본값 0.1은 비현실적으로 느림)
    model = build_model(coreset_ratio=0.01)
    engine = Engine(max_epochs=1)
    train_dm = make_datamodule(test_augmentations=None)
    engine.fit(model=model, datamodule=train_dm)

    print("\n[2/3] 조건별 평가 (조명 7단계 + 각도 5단계, baseline은 둘 다에서 공유)")
    rows = []
    predictions_by_condition = {}
    transform_by_condition = {"baseline": None}

    row, preds = evaluate(model, engine, None, "baseline")
    rows.append(row)
    predictions_by_condition["baseline"] = preds

    for factor in BRIGHTNESS_LEVELS:
        if factor == 1.0:
            continue
        name = f"brightness_{factor}"
        transform = brightness_transform(factor)
        row, preds = evaluate(model, engine, transform, name)
        rows.append(row)
        predictions_by_condition[name] = preds
        transform_by_condition[name] = transform

    for angle in ROTATION_ANGLES:
        if angle == 0:
            continue
        name = f"rotation_{angle}"
        transform = rotation_transform(angle)
        row, preds = evaluate(model, engine, transform, name)
        rows.append(row)
        predictions_by_condition[name] = preds
        transform_by_condition[name] = transform

    save_metrics_csv(rows, RESULTS_DIR / "metrics.csv")

    print("\n[3/3] 시각화")
    by_name = {r["condition"]: r for r in rows}

    brightness_rows = [by_name[f"brightness_{f}"] if f != 1.0 else by_name["baseline"] for f in BRIGHTNESS_LEVELS]
    plot_degradation_curve(
        BRIGHTNESS_LEVELS,
        [r["image_AUROC"] for r in brightness_rows],
        [r["PRO"] for r in brightness_rows],
        xlabel="밝기 배율 (1.0 = 원본)",
        title=f"{CATEGORY}: 밝기 변화에 따른 성능 저하",
        out_path=RESULTS_DIR / "degradation_curve_brightness.png",
    )

    rotation_rows = [by_name[f"rotation_{a}"] if a != 0 else by_name["baseline"] for a in ROTATION_ANGLES]
    plot_degradation_curve(
        ROTATION_ANGLES,
        [r["image_AUROC"] for r in rotation_rows],
        [r["PRO"] for r in rotation_rows],
        xlabel="회전 각도 (도)",
        title=f"{CATEGORY}: 회전 각도에 따른 성능 저하",
        out_path=RESULTS_DIR / "degradation_curve_rotation.png",
    )

    # 각 축에서 image_AUROC가 가장 낮은(가장 많이 깨진) 조건을 골라 시각화
    worst_brightness = min(
        (r for r in rows if r["condition"].startswith("brightness_")),
        key=lambda r: r["image_AUROC"],
    )
    worst_rotation = min(
        (r for r in rows if r["condition"].startswith("rotation_")),
        key=lambda r: r["image_AUROC"],
    )
    print(f"가장 저하가 큰 조명 조건: {worst_brightness['condition']} (AUROC={worst_brightness['image_AUROC']:.4f})")
    print(f"가장 저하가 큰 각도 조건: {worst_rotation['condition']} (AUROC={worst_rotation['image_AUROC']:.4f})")

    save_visual_grid(
        predictions_by_condition[worst_brightness["condition"]],
        worst_brightness["condition"],
        transform_by_condition[worst_brightness["condition"]],
        RESULTS_DIR / f"visual_grid_{worst_brightness['condition']}.png",
    )
    save_visual_grid(
        predictions_by_condition[worst_rotation["condition"]],
        worst_rotation["condition"],
        transform_by_condition[worst_rotation["condition"]],
        RESULTS_DIR / f"visual_grid_{worst_rotation['condition']}.png",
    )

    print("\n완료. results/ 폴더를 확인하세요.")


if __name__ == "__main__":
    main()
