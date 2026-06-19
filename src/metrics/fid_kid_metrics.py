from torch_fidelity import calculate_metrics
import torch
from typing import ClassVar
from src.service.types import MetricKind
from pathlib import Path
import warnings
from src.utils.logger import logger
from src.config import CONFIG
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

class FIDKIDService:
    KIND: ClassVar[MetricKind] = "distribution"
    def __init__(self, device: str | None = None, batch_size: int = 64):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.min_images = 1000
        self.ext= CONFIG['data']['ext']

    def name(self):
        return "fid_kid"

    def compute(self, gt_dir: str, pred_dir: str):
        n_gt, n_pred = [
            sum(1 for f in Path(p).iterdir() if f.suffix.lower() in self.ext)
            for p in (gt_dir, pred_dir)
            ]

        if (n_gt < self.min_images or n_pred < self.min_images) and not CONFIG['fid_kid']['force_run']:
            logger.error(f"Skipping FID/KID — requires ≥{self.min_images}")
            return {
                "fid": float("nan"),
                "kid": float("nan"),
            }
        
        kid_subset_size = min(self.min_images, n_gt, n_pred)
        results = calculate_metrics(
            input1=gt_dir,
            input2=pred_dir,
            fid=True,      
            kid=True,  
            cuda=(self.device == "cuda"),
            batch_size=self.batch_size,
            verbose=False,
            kid_subset_size=kid_subset_size
        )

        return {
            "fid": float(results["frechet_inception_distance"]),
            "kid": float(results["kernel_inception_distance_mean"]),
        }
    