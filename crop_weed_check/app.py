import os
import sys
import time
import base64
import threading
import webbrowser

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

from pipeline.segmentation import segment_furrow
from pipeline.detection import detect_plants
from pipeline.decision import judge_treatment_targets, summarize

PORT = 5050
CONF_THRESHOLD = 0.3

app = Flask(__name__)


def annotate(bgr_image, furrow_mask, judged_plants):
    """고랑 마스크(반투명 초록) + 식물체 박스(초록=crop, 노랑=고랑밖 weed, 빨강=방제대상)를 겹쳐 그림."""
    overlay = bgr_image.copy()
    green_layer = np.zeros_like(bgr_image)
    green_layer[:, :] = (0, 200, 60)
    mask_bool = furrow_mask > 0
    overlay[mask_bool] = cv2.addWeighted(bgr_image, 0.6, green_layer, 0.4, 0)[mask_bool]

    for p in judged_plants:
        x1, y1, x2, y2 = p["box"]
        if p["treatment_target"]:
            color = (0, 0, 255)     # 빨강: 방제 대상 (고랑 안 + weed)
        elif p["label"] == "crop":
            color = (0, 200, 60)    # 초록: 작물
        else:
            color = (0, 210, 255)   # 노랑: 고랑 밖 잡초 (무시 대상)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        tag = f"{p['label']} {p['confidence']:.2f}"
        cv2.putText(overlay, tag, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return overlay


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "이미지 파일이 없습니다."}), 400

    file = request.files["image"]
    try:
        pil_image = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "이미지를 읽을 수 없습니다. 다른 파일을 시도해주세요."}), 400

    bgr_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    t0 = time.time()
    try:
        furrow_mask = segment_furrow(bgr_image)
        plants, detector_mode = detect_plants(bgr_image, CONF_THRESHOLD)
        judged = judge_treatment_targets(furrow_mask, plants)
    except Exception as e:
        return jsonify({"error": f"분석 중 오류가 발생했습니다: {e}"}), 500
    elapsed = time.time() - t0
    fps = round(1.0 / elapsed, 2) if elapsed > 0 else 0.0

    annotated = annotate(bgr_image, furrow_mask, judged)
    ok, buf = cv2.imencode(".png", annotated)
    if not ok:
        return jsonify({"error": "결과 이미지를 생성하지 못했습니다."}), 500
    img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    summary = summarize(judged)
    summary["fps"] = fps
    summary["detector_mode"] = detector_mode  # "yolo" 또는 "rule_based"

    return jsonify({
        "summary": summary,
        "plants": judged,
        "image": f"data:image/png;base64,{img_b64}"
    })


def open_browser():
    time.sleep(1.2)
    try:
        webbrowser.open(f"http://localhost:{PORT}")
    except Exception:
        pass


if __name__ == "__main__":
    print("=" * 50)
    print(" 서버가 실행되었습니다.")
    print(f" 아래 주소를 클릭하거나 브라우저에 복사해서 열어주세요:")
    print(f" http://localhost:{PORT}")
    print(" 종료하려면 터미널에서 Ctrl+C 를 누르세요.")
    print("=" * 50)
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False)
