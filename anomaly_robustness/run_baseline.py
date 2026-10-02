"""
1단계: 제공된 기본 코드를 그대로 실행해 hazelnut 카테고리의 baseline AUROC/F1을 확인한다.

새로 구현하지 않고, 06_PoseEstimation 등과 같은 급의 독립 실습 폴더인 이 프로젝트에서
17_Anomoly_Detection/patchcore_anomaly_detection.py 가 이미 갖고 있는 run_train() 함수를
그대로 가져다 쓴다 (--category 만 hazelnut으로 바꿔서 호출).
이 baseline 수치가 이후 조명·각도 강건성 실험에서 비교할 기준점이 된다.
"""
import os
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
base_code_dir = script_dir.parent / "17_Anomoly_Detection"
sys.path.insert(0, str(base_code_dir))
from patchcore_anomaly_detection import build_parser, run_train  # noqa: E402

# patchcore_anomaly_detection.py는 import되는 순간 자기 자신의 위치(17_Anomoly_Detection/)로
# os.chdir 해버린다. 그대로 두면 체크포인트가 원본 폴더 밑에 생겨서 우리가 새로 만든
# 실험 코드와 뒤섞인다. 데이터셋 경로(Dataset_dir)는 이미 절대경로로 고정된 뒤이므로,
# 여기서 다시 anomaly_robustness/ 로 되돌려서 결과물이 이 폴더 안에만 생기게 한다.
os.chdir(script_dir)

if __name__ == "__main__":
    parser = build_parser()
    # coreset-ratio를 기본값(0.1)보다 낮춘다: CPU에서는 기본값 그대로 쓰면
    # coreset 그리디 선택(최근접 거리 갱신을 수만 번 반복)이 1시간 넘게 걸려서
    # 이 강건성 실험처럼 조건을 여러 번 돌려야 하는 상황에 비현실적이었다.
    # 0.01로 낮추면 메모리 뱅크에 남는 패치 수가 10분의 1로 줄어 수 분 내로 끝난다
    # (절대 정확도는 약간 떨어질 수 있지만, 이 실험의 목적은 "조건 간 상대적
    # 성능 저하 경향"을 보는 것이라 메모리 뱅크 크기 자체보다 일관성이 더 중요하다).
    # num-workers도 0으로: Windows에서는 DataLoader 워커 프로세스를 띄울 때마다
    # 전체 모듈을 재import해서(로그에 같은 경고가 워커 수만큼 중복 출력되는 게 그 증거),
    # 이번처럼 이미지 수가 적은 데이터셋에서는 워커 생성 오버헤드가 오히려 더 크다.
    args = parser.parse_args([
        "train", "--category", "hazelnut",
        "--coreset-ratio", "0.01", "--num-workers", "0",
    ])
    run_train(args)
