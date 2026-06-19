from PIL import Image
import torch
import torchvision.transforms as T
import lpips
from src.metrics.base_metric import MetricService
from typing import ClassVar
from src.service.types import MetricKind

class LPIPSService(MetricService):
    KIND: ClassVar[MetricKind] = "pairwise"
    def __init__(self, net: str = "vgg", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = lpips.LPIPS(net=net).to(self.device).eval()
        self.to_tensor = T.Compose([
            T.ToTensor(),                       
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def name(self):
        return "lpips"

    def _prep(self, img: Image.Image):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.to_tensor(img).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def compute(self, gt_img: Image.Image, pred_img: Image.Image):
        gt = self._prep(gt_img)
        pr = self._prep(pred_img)
        score = self.loss_fn(gt, pr)  
        return float(score.detach().cpu().squeeze().item())
