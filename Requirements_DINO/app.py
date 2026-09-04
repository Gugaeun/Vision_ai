import os
import sys
import io
import time
import base64
import threading
import webbrowser

from PIL import Image, ImageDraw
from flask import Flask, request, jsonify, render_template

#=================================================================
# 초기값 설정
#=================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

PORT = 5000
MODEL_ID = "IDEA-Research/grounding-dino-base"   # 더 빠른 결과가 필요하면 "IDEA-Research/grounding-dino-tiny"
DEFAULT_BOX_THRESHOLD = 0.35
DEFAULT_TEXT_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.5  # 같은 라벨끼리 이 값 이상 겹치면 낮은 점수 박스를 제거
MAX_BOX_AREA_RATIO = 0.85  # 박스가 이미지 면적의 이 비율 이상이면 "전체를 통째로 덮는" 오탐으로 보고 제거
BOX_COLOR = (242, 183, 5)  # 결과 이미지에 그릴 박스 색상 (앰버)

app = Flask(__name__)

#=================================================================
# 모델 로드
# Grounding DINO는 프롬프트 "클래스 목록"이 아니라 문장 전체를 텍스트로 받아,
# 이미지와 텍스트를 함께 이해해서 매칭되는 영역을 찾는 방식(phrase grounding)이다.
#=================================================================
print("모델을 불러오는 중입니다. 처음 실행 시 가중치 파일(수백MB)을 내려받아 시간이 걸릴 수 있습니다...")
try:
	import torch
	from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

	device = "cuda" if torch.cuda.is_available() else "cpu"
	processor = AutoProcessor.from_pretrained(MODEL_ID)
	model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
	model.eval()
	print(f"모델 로드 완료. (device={device})")
except Exception as e:
	print("Error: 모델을 불러오지 못했습니다.")
	print(e)
	sys.exit(1)


def build_prompt_text(prompts):
	# Grounding DINO 권장 형식: 소문자, 구문마다 마침표로 구분. 예) "rotten fruit. dented can."
	cleaned = [p.strip().rstrip(".").lower() for p in prompts if p.strip()]
	return ". ".join(cleaned) + "."


def box_iou(box_a, box_b):
	ax1, ay1, ax2, ay2 = box_a
	bx1, by1, bx2, by2 = box_b
	inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
	inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
	inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
	area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
	area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
	union = area_a + area_b - inter_area
	return inter_area / union if union > 0 else 0.0


def apply_nms(detections, iou_threshold=NMS_IOU_THRESHOLD):
	# 임계값을 낮게 잡을수록 같은 물체에 여러 박스가 중복으로 잡히므로,
	# 라벨과 상관없이 많이 겹치는 박스는 신뢰도가 낮은 쪽을 제거한다(클래스 무관 NMS).
	ordered = sorted(detections, key=lambda d: d["confidence"], reverse=True)
	kept = []
	for cand in ordered:
		if any(box_iou(cand["box"], k["box"]) > iou_threshold for k in kept):
			continue
		kept.append(cand)
	return kept


def run_detection(pil_image, prompts, box_threshold, text_threshold):
	w, h = pil_image.size
	image_area = w * h

	# 프롬프트를 한 문장으로 합쳐서 넣으면, 임계값이 낮을 때 서로 다른 문구가
	# 마침표 경계를 넘어 하나의 라벨로 섞여버린다(예: "rotten fruit moldy vegetable dented can").
	# 이를 막기 위해 프롬프트마다 모델을 따로 호출한다.
	detections = []
	for prompt in prompts:
		text = build_prompt_text([prompt])
		inputs = processor(images=pil_image, text=text, return_tensors="pt").to(device)
		with torch.no_grad():
			outputs = model(**inputs)

		results = processor.post_process_grounded_object_detection(
			outputs,
			input_ids=inputs.input_ids,
			threshold=box_threshold,
			text_threshold=text_threshold,
			target_sizes=[(h, w)]
		)[0]

		for box, score in zip(results["boxes"], results["scores"]):
			x1, y1, x2, y2 = [round(float(v), 1) for v in box.tolist()]
			box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
			if image_area > 0 and box_area / image_area > MAX_BOX_AREA_RATIO:
				continue  # 사진 전체를 통째로 덮는 박스는 실제 이상 항목이 아닐 가능성이 높음
			detections.append({
				"label": prompt,
				"confidence": round(float(score), 3),
				"box": [x1, y1, x2, y2]
			})
	detections = apply_nms(detections)
	detections.sort(key=lambda d: d["confidence"], reverse=True)

	annotated = pil_image.copy()
	draw = ImageDraw.Draw(annotated)
	line_w = max(2, int(w * 0.004))
	for d in detections:
		x1, y1, x2, y2 = d["box"]
		draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=line_w)
		draw.text((x1, max(0, y1 - 16)), f'{d["label"]} {d["confidence"]}', fill=BOX_COLOR)

	return annotated, detections


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
		box_threshold = float(request.form.get("box_threshold", DEFAULT_BOX_THRESHOLD))
	except ValueError:
		box_threshold = DEFAULT_BOX_THRESHOLD
	try:
		text_threshold = float(request.form.get("text_threshold", DEFAULT_TEXT_THRESHOLD))
	except ValueError:
		text_threshold = DEFAULT_TEXT_THRESHOLD

	prompts = [p.strip() for p in prompts_raw.split(",") if p.strip()]
	if not prompts:
		return jsonify({"error": "탐지할 프롬프트를 1개 이상 입력해주세요."}), 400

	try:
		pil_image = Image.open(file.stream).convert("RGB")
	except Exception:
		return jsonify({"error": "이미지를 읽을 수 없습니다. 다른 파일을 시도해주세요."}), 400

	try:
		annotated, detections = run_detection(pil_image, prompts, box_threshold, text_threshold)
	except Exception as e:
		return jsonify({"error": f"탐지 중 오류가 발생했습니다: {e}"}), 500

	buf = io.BytesIO()
	annotated.save(buf, format="PNG")
	img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

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
	print(" (CPU 환경에서는 이미지 1장 분석에 수 초~수십 초가 걸릴 수 있습니다)")
	print(" 종료하려면 터미널에서 Ctrl+C 를 누르세요.")
	print("=" * 50)
	threading.Thread(target=open_browser, daemon=True).start()
	app.run(host="0.0.0.0", port=PORT, debug=False)
