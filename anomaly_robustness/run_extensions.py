"""
확장 실험 3가지 (슬라이드 "한 단계 더 나아가기" 중 지금 바로 할 수 있는 것들):

  1) 코어셋 비율별 속도·정확도 곡선 — ratio를 바꿔가며 baseline을 재학습해서
     "memory bank를 얼마나 작게 줄여도 되는가"를 재본다.
  2) 미검출·과검출 사례 분석표 — baseline과 가장 저하가 컸던 두 조건(brightness_1.6,
     rotation_45)에서 모델이 틀린 이미지만 모아 표로 정리한다.
  3) 결함 유형별 검출률 — hazelnut의 결함 4종(crack/cut/hole/print)별로 baseline에서
     얼마나 잘 잡는지 따로 집계한다 (파일 경로의 폴더명으로 유형을 구분한다).

실행: python run_extensions.py
"""
import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))
base_code_dir = script_dir.parent / "17_Anomoly_Detection"
sys.path.insert(0, str(base_code_dir))

from patchcore_anomaly_detection import build_model  # noqa: E402
from anomalib.engine import Engine  # noqa: E402
from anomalib.metrics import AUROC, PRO  # noqa: E402
from run_experiment import CATEGORY, RESULTS_DIR, make_datamodule  # noqa: E402
from perturbations import brightness_transform, rotation_transform  # noqa: E402


def defect_type_from_path(path: str) -> str:
    """.../test/<결함종류>/000.png 또는 .../test/good/000.png 에서 결함종류(또는 good)를 뽑는다."""
    parts = Path(path).parts
    idx = parts.index("test") if "test" in parts else -2
    return parts[idx + 1]


# ----------------------------------------------------------------------
# 1) 코어셋 비율별 속도·정확도
# ----------------------------------------------------------------------
def run_coreset_sweep(ratios=(0.01, 0.02, 0.05)):
    print(f"\n[1] 코어셋 비율 스윕: {ratios}")
    rows = []
    for ratio in ratios:
        model = build_model(coreset_ratio=ratio)
        engine = Engine(max_epochs=1, enable_progress_bar=False)
        t0 = time.time()
        engine.fit(model=model, datamodule=make_datamodule(test_augmentations=None))
        fit_time = time.time() - t0

        predictions = engine.predict(model=model, datamodule=make_datamodule(None), return_predictions=True)
        auroc_metric = AUROC(fields=["pred_score", "gt_label"])
        pro_metric = PRO(fields=["anomaly_map", "gt_mask"])
        for batch in predictions:
            auroc_metric.update(batch)
            pro_metric.update(batch)
        auroc = float(auroc_metric.compute())
        pro = float(pro_metric.compute())

        print(f"  ratio={ratio}  fit_time={fit_time:.1f}s  AUROC={auroc:.4f}  PRO={pro:.4f}")
        rows.append({"coreset_ratio": ratio, "fit_time_sec": round(fit_time, 1),
                     "image_AUROC": auroc, "PRO": pro})

    with open(RESULTS_DIR / "coreset_sweep.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["coreset_ratio", "fit_time_sec", "image_AUROC", "PRO"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"저장됨: {RESULTS_DIR / 'coreset_sweep.csv'}")

    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    ratios_x = [r["coreset_ratio"] for r in rows]
    ax1.plot(ratios_x, [r["fit_time_sec"] for r in rows], marker="o", color="#8a5a2a", label="학습 시간(초)")
    ax1.set_xlabel("coreset_sampling_ratio")
    ax1.set_ylabel("학습 시간 (초)", color="#8a5a2a")
    ax1.tick_params(axis="y", labelcolor="#8a5a2a")

    ax2 = ax1.twinx()
    ax2.plot(ratios_x, [r["image_AUROC"] for r in rows], marker="s", color="#2f8f5c", label="Image AUROC")
    ax2.plot(ratios_x, [r["PRO"] for r in rows], marker="^", color="#2f6f9f", label="PRO")
    ax2.set_ylabel("AUROC / PRO")
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)
    ax1.set_title(f"{CATEGORY}: 코어셋 비율에 따른 속도·정확도 트레이드오프")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "coreset_sweep.png", dpi=150)
    plt.close(fig)
    print(f"저장됨: {RESULTS_DIR / 'coreset_sweep.png'}")
    return rows


# ----------------------------------------------------------------------
# 2) 미검출·과검출 사례 + 3) 결함 유형별 검출률
# ----------------------------------------------------------------------
def run_error_and_defect_analysis():
    print("\n[2/3] 미검출·과검출 사례 + 결함 유형별 검출률 (baseline 모델 재사용)")
    model = build_model(coreset_ratio=0.01)
    engine = Engine(max_epochs=1, enable_progress_bar=False)
    engine.fit(model=model, datamodule=make_datamodule(test_augmentations=None))

    conditions = {
        "baseline": None,
        "brightness_1.6": brightness_transform(1.6),
        "rotation_45": rotation_transform(45),
    }

    error_rows = []
    defect_rows = []
    for name, transform in conditions.items():
        predictions = engine.predict(model=model, datamodule=make_datamodule(transform), return_predictions=True)

        # 결함 유형별 집계용 카운터: {유형: [총개수, 정탐개수]}
        per_type = {}
        for batch in predictions:
            paths = batch.image_path
            gts = batch.gt_label.tolist()
            preds = batch.pred_label.tolist()
            scores = batch.pred_score.tolist()
            for path, gt, pred, score in zip(paths, gts, preds, scores):
                dtype = defect_type_from_path(path)
                per_type.setdefault(dtype, [0, 0])
                per_type[dtype][0] += 1
                is_correct = (gt == pred)
                if is_correct:
                    per_type[dtype][1] += 1
                if not is_correct:
                    error_rows.append({
                        "condition": name,
                        "defect_type": dtype,
                        "file": Path(path).name,
                        "gt": "결함" if gt == 1 else "정상",
                        "pred": "결함" if pred == 1 else "정상",
                        "error_kind": "미검출" if (gt == 1 and pred == 0) else "과검출",
                        "score": round(float(score), 4),
                    })

        for dtype, (total, correct) in sorted(per_type.items()):
            defect_rows.append({
                "condition": name,
                "defect_type": dtype,
                "total": total,
                "correct": correct,
                "accuracy": round(correct / total, 4) if total else float("nan"),
            })
        print(f"  [{name}] 유형별: " + ", ".join(f"{d}={c}/{t}" for d, (t, c) in sorted(per_type.items())))

    with open(RESULTS_DIR / "error_cases.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["condition", "defect_type", "file", "gt", "pred", "error_kind", "score"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(error_rows)
    print(f"저장됨: {RESULTS_DIR / 'error_cases.csv'} ({len(error_rows)}건)")

    with open(RESULTS_DIR / "defect_type_accuracy.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["condition", "defect_type", "total", "correct", "accuracy"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(defect_rows)
    print(f"저장됨: {RESULTS_DIR / 'defect_type_accuracy.csv'}")

    return error_rows, defect_rows


def main():
    run_coreset_sweep()
    run_error_and_defect_analysis()
    print("\n완료.")


if __name__ == "__main__":
    main()
