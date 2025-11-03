from src.utils.logger import logger
from src.service.image_loader_service import ImageLoaderService
from src.metrics.ssim_metrics import SSIMService
from src.utils.image_pair_utils import ImagePairUtil
from pathlib import Path
import numpy as np
from src.config import CONFIG

def main(gt_dir: Path, pred_dir: Path, is_skipped: bool = False):
    loader = ImageLoaderService(gt_dir=gt_dir, pred_dir=pred_dir,resize=(512, 512))

    print(f"Found {len(loader.gt_index)} GT images")
    print(f"Found {len(loader.pred_index)} Pred images")
    
    check_alignment(loader.check_alignment()) if not is_skipped else logger.info("Skipped alignment check")

    metrics = [SSIMService()]

    metric_values = {m.name(): [] for m in metrics}
    for fname, gt_img, pred_img in loader.iter_pairs():
        logger.info(f"Evaluating: {fname}")
        for metric in metrics:
            value = metric.compute(gt_img, pred_img)
            metric_values[metric.name()].append(value)
            logger.info(f"{metric.name().upper()}: {value:.4f}")

    logger.info("=== SUMMARY ===")
    for name, values in metric_values.items():
        mean_val = np.mean(values)
        logger.info(f"{name.upper()} mean: {mean_val:.4f}")

def check_alignment(align):
    assert not align["missing_predictions"], f"Missing predictions: {align['missing_predictions']}"
    assert not align["missing_ground_truth"], f"Missing GTs: {align['missing_ground_truth']}"

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent
    GT_DIR = (ROOT / CONFIG['paths']['gt_dir']).resolve()
    PRED_DIR = (ROOT / CONFIG['paths']['pred_dir']).resolve()
    main(GT_DIR, PRED_DIR,True)