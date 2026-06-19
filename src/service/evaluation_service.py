from unicodedata import name
from src.utils.logger import logger
from src.service.image_loader_service import ImageLoaderService
from src.metrics.ssim_metrics import SSIMService
from src.metrics.lpips_metrics import LPIPSService
from src.metrics.clip_metrics import CLIPImageSimilarityService
# from src.metrics.fid_kid_metrics import FIDKIDService
from src.metrics.fid_kid_metrics_v2 import FIDKIDService

from tqdm import tqdm 
from src.utils.image_pair_utils import ImagePairUtil
from pathlib import Path
import numpy as np
from src.config import CONFIG

def main(gt_dir: Path, pred_dir_self: Path, pred_dir_pair: Path, out_dir: Path, is_skipped: bool = False):
    loader_self = ImageLoaderService(gt_dir=gt_dir, pred_dir=pred_dir_self)
    loader_pair = ImageLoaderService(gt_dir=gt_dir, pred_dir=pred_dir_pair)

    if len(loader_self.gt_index) == 0 or len(loader_self.pred_index) == 0 or len(loader_pair.pred_index) == 0:
        raise ValueError(
            f"No valid images found. GT={len(loader_self.gt_index)}, Pred_Self={len(loader_self.pred_index)}, Pred_Pair={len(loader_pair.pred_index)}"
        )

    logger.info(f"Found {len(loader_self.gt_index)} GT images")
    logger.info(f"Found {len(loader_self.pred_index)} Pred_Self images")
    logger.info(f"Found {len(loader_pair.pred_index)} Pred_Pair images")
    
    check_alignment(loader_self.check_alignment()) if not is_skipped else logger.info("Skipped alignment check")

    metrics = [SSIMService(),LPIPSService(),CLIPImageSimilarityService(),FIDKIDService()]

    results = {}
    for metric in metrics:
        r = evaluate(metric, loader_self, loader_pair)
        results.update(r)

    summary(results, out_dir)

def check_alignment(align):
    assert not align["missing_predictions"], f"Missing predictions: {align['missing_predictions']}"
    assert not align["missing_ground_truth"], f"Missing GTs: {align['missing_ground_truth']}"

def evaluate(metric, loader_self, loader_pair):
    results = {}
    name = metric.name()
    kind = getattr(metric, "KIND", "pairwise")

    logger.info(f"Using metric: {metric.__class__.__name__} [KIND={kind}]")

    if kind == "pairwise":
        pairs = list(loader_self.iter_pairs())
        vals = []

        for fname, gt_img, pred_img in tqdm(pairs, desc=f"Evaluating {name.upper()}"):
            v = float(metric.compute(gt_img, pred_img))
            vals.append(v)

        mean_val = float(np.mean(vals)) if vals else float("nan")
        results[name] = mean_val
        logger.info(f"{name.upper()} mean: {mean_val:.6f}")
        return results
    else:
        logger.info(f"--- Computing {name.upper()} [PAIRED] ---")
        res_paired = metric.compute(str(loader_self.gt_dir), str(loader_self.pred_dir))
        for k, v in res_paired.items():
            results[f"{k}_paired"] = v
            
        logger.info(f"--- Computing {name.upper()} [UNPAIRED] ---")
        res_unpaired = metric.compute(str(loader_pair.gt_dir), str(loader_pair.pred_dir))
        for k, v in res_unpaired.items():
            results[f"{k}_unpaired"] = v

        log_msg = f"{name.upper()}: " + ", ".join(f"{k}={v:.6f}" for k, v in results.items())
        logger.info(log_msg)
        
        return results

def summary(results, out_dir: Path):
    header = "======================== SUMMARY ========================"
    logger.info(header)
    lines = [header + "\n"]
    
    for k, v in results.items():
        if isinstance(v, dict):
            msg = f"{k}: " + ", ".join(f"{kk}={vv:.6f}" for kk, vv in v.items())
        else:
            msg = f"{k}: {v:.6f}"
        logger.info(msg)
        lines.append(msg + "\n")

    if out_dir is None:
        logger.info("Skipping summary file writing.")
        return  

    try:
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        out_file = out_dir / "summary.txt"
        out_file.write_text("".join(lines), encoding="utf-8")
        logger.info(f"Summary saved to: {out_file}")

    except Exception as e:
        logger.error(f"Failed to write summary file: {e}")

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent
    GT_DIR = (ROOT / CONFIG['paths']['gt_dir']).resolve()
    PRED_DIR_SELF = (ROOT / CONFIG['paths']['pred_dir_self']).resolve()
    PRED_DIR_PAIR = (ROOT / CONFIG['paths']['pred_dir_pair']).resolve()
    OUT_DIR = (None if not CONFIG["paths"].get("output_dir") else (ROOT / CONFIG["paths"]["output_dir"]).resolve())
    main(GT_DIR, PRED_DIR_SELF, PRED_DIR_PAIR, OUT_DIR, True)