"""
조명(밝기)·각도(회전) 변형 함수 모음.

anomalib의 MVTecAD 데이터모듈은 `test_augmentations` 인자로 평가용 이미지에만
적용할 추가 변형을 받는다. 여기서 만드는 변형들은 그 자리에 꽂아서 쓴다
(학습(memory bank 구축)에는 영향을 주지 않고, 평가 시점의 테스트 이미지만 바뀐다).

주의: anomalib 내부에서는 이미지를 PIL이 아니라 torch 텐서(C,H,W, uint8)로 읽고,
세그멘테이션 태스크에서는 `augmentations(image_tensor, mask_tensor)` 형태로
이미지·마스크를 같이 넘겨서 같이 변형하라고 요구한다 (회전처럼 기하학적 변형일 때
이미지와 결함 마스크 위치가 어긋나면 안 되기 때문). 그래서 아래 변형들은 PIL이 아닌
torch 텐서를 직접 받아 처리한다.
"""
import cv2
import numpy as np
import torch


def _tensor_to_bgr(img: torch.Tensor) -> np.ndarray:
    """(C,H,W) uint8 텐서 -> cv2가 다루는 (H,W,C) ndarray."""
    return img.permute(1, 2, 0).cpu().numpy()


def _bgr_to_tensor(arr: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).to(like.dtype)


class AdjustBrightness:
    """밝기 배율을 곱하는 변형. factor=1.0이면 원본 그대로. 마스크는 손대지 않는다."""

    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, img: torch.Tensor, mask=None):
        if self.factor == 1.0:
            return img, mask
        arr = img.to(torch.float32) * self.factor
        img = arr.clamp(0, 255).to(img.dtype)
        return img, mask

    def __repr__(self):
        return f"AdjustBrightness(factor={self.factor})"


class RotateReflect:
    """
    고정 각도로 이미지와 마스크를 같이 회전시키는 변형.

    회전하면 이미지 모서리에 빈 공간이 생기는데, 검은색(constant)으로 채우면
    그 자체가 "배경과 다른 영역"이라 이상 탐지 모델이 그 모서리를 결함으로
    오탐할 수 있다 (스펙의 "자주 막히는 지점" 항목). 그래서 이미지는
    cv2.warpAffine의 BORDER_REFLECT_101로 가장자리를 거울처럼 비춰 채워서
    급격한 색 경계가 생기지 않게 한다. 마스크는 반사시키면 없던 결함이
    가장자리에 복제돼 보일 수 있어서, 빈 공간을 0(정상)으로 채운다.
    """

    def __init__(self, angle: float):
        self.angle = angle

    def _rotate(self, arr: np.ndarray, border_mode: int, border_value=0) -> np.ndarray:
        h, w = arr.shape[:2]
        center = (w / 2, h / 2)
        rot_mat = cv2.getRotationMatrix2D(center, self.angle, 1.0)
        return cv2.warpAffine(arr, rot_mat, (w, h), borderMode=border_mode, borderValue=border_value)

    def __call__(self, img: torch.Tensor, mask=None):
        if self.angle == 0:
            return img, mask

        img_arr = _tensor_to_bgr(img)
        rotated_img = self._rotate(img_arr, cv2.BORDER_REFLECT_101)
        img = _bgr_to_tensor(rotated_img, img)

        if mask is not None:
            mask_arr = mask.cpu().numpy()
            rotated_mask = self._rotate(mask_arr, cv2.BORDER_CONSTANT, border_value=0)
            mask = torch.from_numpy(np.ascontiguousarray(rotated_mask)).to(mask.dtype)
        return img, mask

    def __repr__(self):
        return f"RotateReflect(angle={self.angle})"


# 밝기: 1.0(원본, baseline)을 중심으로 어둡게/밝게 양방향으로 단계를 둔다.
BRIGHTNESS_LEVELS = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]

# 각도: 스펙 권장값 그대로. 0도가 baseline이다.
ROTATION_ANGLES = [0, 5, 10, 20, 45]


def brightness_transform(factor: float) -> AdjustBrightness:
    return AdjustBrightness(factor)


def rotation_transform(angle: float) -> RotateReflect:
    return RotateReflect(angle)
