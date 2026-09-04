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

#=================================================================
# 초기값 설정
#=================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

PORT = 5000
MODEL_NAME = "yolov8x-worldv2.pt"   # ultralytics가 최초 실행 시 자동으로 내려받음 (s보다 훨씬 정확하지만 무겁고 느림)
DEFAULT_CONF = 0.3  # yolov8x-worldv2 기준: 이 사진에서 정답 신뢰도가 0.3~0.4대라 너무 낮추면 오탐이 늚

app = Flask(__name__)

#=================================================================
# 모델 로드 (OVD: 텍스트 프롬프트로 클래스를 즉석에서 바꿀 수 있는 탐지기)
#=================================================================
print("모델을 불러오는 중입니다. 처음 실행 시 가중치 파일을 내려받아 시간이 걸릴 수 있습니다...")
try:
	import torch
	from ultralytics import YOLOWorld

	device = "cuda" if torch.cuda.is_available() else "cpu"
	model = YOLOWorld(MODEL_NAME)
	model.to(device)
	print(f"모델 로드 완료. (device={device})")
except Exception as e:
	print("Error: 모델을 불러오지 못했습니다.")
	print(e)
	sys.exit(1)


def run_detection(pil_image, prompts, conf_threshold):
	"""
	OVD 탐지 실행.
	prompts: ["rotten fruit", "damaged packaging", ...] 형태의 영어 프롬프트 리스트
	반환: (annotated_bgr_ndarray, detections_list)
	"""
	# 프롬프트를 클래스로 즉석 등록 (재학습 없이 새 개념 추가 가능한 부분)
	model.set_classes(prompts)

	results = model.predict(pil_image, conf=conf_threshold, verbose=False)
	r = results[0]

	detections = []
	for box in r.boxes:
		cls_id = int(box.cls[0])
		label = prompts[cls_id] if cls_id < len(prompts) else str(cls_id)
		conf = float(box.conf[0])
		x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].tolist()]
		detections.append({
			"label": label,
			"confidence": round(conf, 3),
			"box": [x1, y1, x2, y2]
		})

	# 신뢰도 높은 순으로 정렬
	detections.sort(key=lambda d: d["confidence"], reverse=True)

	annotated_bgr = r.plot()  # 바운딩 박스가 그려진 BGR 이미지(numpy)
	return annotated_bgr, detections


#=================================================================
# 라우트
#=================================================================
@app.route("/")
def index():
	return render_template("index.html")


@app.route("/api/detect", methods=["POST"])
def detect():
	if "image" not in request.files:
		return jsonify({"error": "이미지 파일이 없습니다."}), 400

	file = request.files["image"]
	prompts_raw = request.form.get("prompts", "")
	try:
		conf_threshold = float(request.form.get("threshold", DEFAULT_CONF))
	except ValueError:
		conf_threshold = DEFAULT_CONF

	prompts = [p.strip() for p in prompts_raw.split(",") if p.strip()]
	if not prompts:
		return jsonify({"error": "탐지할 프롬프트를 1개 이상 입력해주세요."}), 400

	try:
		pil_image = Image.open(file.stream).convert("RGB")
	except Exception:
		return jsonify({"error": "이미지를 읽을 수 없습니다. 다른 파일을 시도해주세요."}), 400

	try:
		annotated_bgr, detections = run_detection(pil_image, prompts, conf_threshold)
	except Exception as e:
		return jsonify({"error": f"탐지 중 오류가 발생했습니다: {e}"}), 500

	ok, buf = cv2.imencode(".png", annotated_bgr)
	if not ok:
		return jsonify({"error": "결과 이미지를 생성하지 못했습니다."}), 500
	img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

	return jsonify({
		"count": len(detections),
		"detections": detections,
		"image": f"data:image/png;base64,{img_b64}"
	})


#=================================================================
# 서버 실행
#=================================================================
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
