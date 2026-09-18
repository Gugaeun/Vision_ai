"""
필수 요건 3: FP32 / FP16 / INT8 중 2개 이상을 같은 영상으로 FPS·자원 사용률 비교.

실행:
    python pipeline/benchmark.py --video sample_data/sample.mp4 --weights models/yolov8_crop_weed.pt

crop/weed 전용 가중치가 아직 없다면 --weights를 생략 시 yolov8n.pt(COCO 사전학습)로 대체해서
파이프라인 자체(정밀도별 export + 벤치마크 절차)를 먼저 검증할 수 있다.
"""
import argparse
import os
import time

import cv2
import psutil


def export_precisions(weights_path, out_dir):
    """FP32(pt 원본) / FP16(onnx, half=True) / INT8(onnx, int8=True) 세 버전을 준비."""
    from ultralytics import YOLO
    os.makedirs(out_dir, exist_ok=True)

    model = YOLO(weights_path)
    paths = {"FP32": weights_path}

    fp16_path = os.path.join(out_dir, "model_fp16.onnx")
    if not os.path.exists(fp16_path):
        exported = model.export(format="onnx", half=True)
        os.replace(exported, fp16_path)
    paths["FP16"] = fp16_path

    int8_path = os.path.join(out_dir, "model_int8.onnx")
    if not os.path.exists(int8_path):
        exported = model.export(format="onnx", int8=True)
        os.replace(exported, int8_path)
    paths["INT8"] = int8_path

    return paths


def benchmark_one(weights_path, video_path, max_frames=150):
    from ultralytics import YOLO
    model = YOLO(weights_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

    proc = psutil.Process(os.getpid())
    proc.cpu_percent(interval=None)  # 기준점 초기화

    n_frames = 0
    t0 = time.time()
    mem_samples = []
    while n_frames < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        model.predict(frame, verbose=False)
        n_frames += 1
        if n_frames % 10 == 0:
            mem_samples.append(proc.memory_info().rss / (1024 * 1024))
    elapsed = time.time() - t0
    cap.release()

    cpu_percent = proc.cpu_percent(interval=None)
    fps = n_frames / elapsed if elapsed > 0 else 0.0
    avg_mem_mb = sum(mem_samples) / len(mem_samples) if mem_samples else proc.memory_info().rss / (1024 * 1024)

    return {"frames": n_frames, "fps": round(fps, 2), "cpu_percent": round(cpu_percent, 1), "avg_mem_mb": round(avg_mem_mb, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--out_dir", default="models")
    parser.add_argument("--max_frames", type=int, default=150)
    args = parser.parse_args()

    print(f"가중치: {args.weights}")
    print("FP16/INT8 버전 export 중 (없으면 새로 생성)...")
    paths = export_precisions(args.weights, args.out_dir)

    print("\n같은 영상으로 정밀도별 벤치마크 실행 중...\n")
    results = {}
    for precision, path in paths.items():
        print(f"[{precision}] {path}")
        try:
            results[precision] = benchmark_one(path, args.video, args.max_frames)
        except Exception as e:
            results[precision] = {"error": str(e)}
        print(f"  -> {results[precision]}")

    print("\n" + "=" * 60)
    print(f"{'정밀도':<8}{'FPS':>10}{'CPU%':>10}{'평균 메모리(MB)':>18}")
    for precision, r in results.items():
        if "error" in r:
            print(f"{precision:<8}{'ERROR':>10}  {r['error']}")
        else:
            print(f"{precision:<8}{r['fps']:>10}{r['cpu_percent']:>10}{r['avg_mem_mb']:>18}")
    print("=" * 60)


if __name__ == "__main__":
    main()
