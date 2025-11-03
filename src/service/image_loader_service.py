from pathlib import Path
from typing import Iterator, Optional, Tuple, List, Dict
from PIL import Image
from src.utils.logger import logger
from src.utils.image_pair_utils import ImagePairUtil

class ImageLoaderService:
    def __init__(self,gt_dir: str,pred_dir: str,exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp"),resize: Optional[Tuple[int, int]] = None):
        self.gt_dir = Path(gt_dir)
        self.pred_dir = Path(pred_dir)
        self.extensions = tuple(e.lower() for e in exts)
        self.resize = resize

        if not self.gt_dir.exists():
            raise FileNotFoundError(f"Ground Truth directory not found: {gt_dir}")
        if not self.pred_dir.exists():
            raise FileNotFoundError(f"Predictions directory not found: {pred_dir}")

        self.gt_index = self._index_dir(self.gt_dir)
        self.pred_index = self._index_dir(self.pred_dir)

    def _index_dir(self, d: Path):
        files = {}
        for p in d.iterdir():
            if p.suffix.lower() in self.extensions:
                files[p.name] = p
        return files

    def _load_img(self, path: Path):
        img = Image.open(path).convert("RGB")
        if self.resize:
            img = img.resize(self.resize, Image.BICUBIC)
        return img

    def iter_pairs(self):
        for gt_path, pred_path in ImagePairUtil.match_and_convert(self.gt_dir, self.pred_dir):
            gt_img = self._load_img(gt_path)
            pred_img = self._load_img(pred_path)
            yield gt_path.stem, gt_img, pred_img

    def check_alignment(self):
        gt_files = set(self.gt_index.keys())
        pred_files = set(self.pred_index.keys())

        gt_no_pred = list(gt_files - pred_files)   
        pred_no_gt = list(pred_files - gt_files)    

        if gt_no_pred:
            logger.error(f"Missing predictions, count: {len(gt_no_pred)}")
            list(map(lambda file_name: print(f"{file_name}"), gt_no_pred))

        if pred_no_gt:
            logger.error(f"Missing Ground Truth, count: {len(pred_no_gt)}")
            list(map(lambda file_name: print(f"{file_name}"), pred_no_gt))

        return {
            "missing_predictions": gt_no_pred,
            "missing_ground_truth": pred_no_gt,
        }