from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from src.metrics.base_metric import MetricService
from typing import ClassVar
from src.service.types import MetricKind
class SSIMService(MetricService):
    KIND: ClassVar[MetricKind] = "pairwise"
    def __init__(self):
        pass

    def name(self):
        return "ssim"

    def _to_numpy(self, img: Image.Image):
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        return arr

    def compute(self, gt_img: Image.Image, pred_img: Image.Image):
        gt = self._to_numpy(gt_img)
        pr = self._to_numpy(pred_img)

        score = ssim(gt, pr, channel_axis=2, data_range=1.0)
        return float(score)