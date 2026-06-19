import torch
from typing import ClassVar
from src.service.types import MetricKind
from pathlib import Path
from src.utils.logger import logger
from src.config import CONFIG
from cleanfid import fid

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

        force_run = CONFIG.get('fid_kid', {}).get('force_run', False)
        if (n_gt < self.min_images or n_pred < self.min_images) and not force_run:
            logger.warning(f"Skipping FID/KID — requires ≥{self.min_images} images. Found {n_pred}.")
            return {"fid": float("nan"), "kid": float("nan")}
        
        try:
            score_fid = fid.compute_fid(
                fdir1=gt_dir, 
                fdir2=pred_dir, 
                mode="clean",         
                device=torch.device(self.device),
                batch_size=self.batch_size,
                num_workers=0,        
                verbose=False
            )
            
            score_kid = fid.compute_kid(
                fdir1=gt_dir, 
                fdir2=pred_dir, 
                mode="clean",         
                device=torch.device(self.device),
                batch_size=self.batch_size,
                num_workers=0,        
                verbose=False
            )
            
        except Exception as e:
            logger.error(f"clean-fid computation failed: {e}")
            return {"fid": float("nan"), "kid": float("nan"), "kid_scaled": float("nan")}

        return {
            "fid": float(score_fid),
            "kid": float(score_kid),
            "kid_scaled": float(score_kid) * 1000.0, 
        }