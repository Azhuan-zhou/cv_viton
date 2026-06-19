from PIL import Image
import torch
import torchvision.transforms as T
import clip 
from src.metrics.base_metric import MetricService
from typing import ClassVar
from src.service.types import MetricKind

class CLIPImageSimilarityService(MetricService):
    KIND: ClassVar[MetricKind] = "pairwise"
    def __init__(self, model_name: str = "ViT-B/32", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    def name(self):
        return "clip_i"

    def _prep(self, img: Image.Image):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.preprocess(img).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def compute(self, gt_img: Image.Image, pred_img: Image.Image):
        gt = self._prep(gt_img)
        pr = self._prep(pred_img)
        e1 = self.model.encode_image(gt)
        e2 = self.model.encode_image(pr)
        e1 = e1 / e1.norm(dim=-1, keepdim=True)
        e2 = e2 / e2.norm(dim=-1, keepdim=True)
        cos = (e1 * e2).sum(dim=-1).item()
        return float(cos)                     
