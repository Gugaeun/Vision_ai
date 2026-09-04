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
MODEL_NAME = "yoloe-11l-seg.pt"   # ultralytics가 최초 실행 시 자동으로 내려받음
DEFAULT_CONF = 0.2

app = Flask(__name__)

#=================================================================
# 모델 로드 (YOLOE: prompt-free/text-prompt 겸용 open-vocabulary 탐지기)
#=================================================================
print("모델을 불러오는 중입니다. 처음 실행 시 가중치 파일을 내려받아 시간이 걸릴 수 있습니다...")
try:
	import torch
	from ultralytics import YOLOE

	device = "cuda" if torch.cuda.is_available() else "cpu"
	model = YOLOE(MODEL_NAME)
	model.to(device)
	print(f"모델 로드 완료. (device={device})")
except Exception as e:
	print("Error: 모델을 불러오지 못했습니다.")
	print(e)
	sys.exit(1)


def run_detection(pil_image, prompts, conf_threshold):
	"""
	YOLOE 탐지 실행.
	prompts: ["rotten fruit", "damaged packaging", ...] 형태의 영어 프롬프트 리스트
	반환: (annotated_bgr_ndarray, detections_list)
	"""
	# 프롬프트를 텍스트 임베딩(PE)으로 변환해 클래스로 즉석 등록
	text_pe = model.get_text_pe(prompts)
	model.set_classes(prompts, text_pe)

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

	# masks=False: yoloe-*-seg 모델은 기본적으로 감지 영역을 반투명 색으로 덮어 그리는데(세그멘테이션),
	# DINO/YOLO-World와 화면을 통일하기 위해 바운딩 박스만 그리게 한다.
	annotated_bgr = r.plot(masks=False)  # 바운딩 박스가 그려진 BGR 이미지(numpy)
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
