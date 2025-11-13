from pathlib import Path
from PIL import Image
from src.utils.logger import logger
from src.config import CONFIG

class ImagePairUtil:
    SUPPORTED_EXTS = tuple(e.lower() for e in CONFIG['data']['ext'])

    @staticmethod
    def match_and_convert(gt_dir: Path, pred_dir: Path):
        pred_index = {
            p.stem.lower(): p for p in pred_dir.iterdir()
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in ImagePairUtil.SUPPORTED_EXTS
        }

        for gt_path in gt_dir.iterdir():
            if not gt_path.is_file():
                logger.warning(f"Skipping non-file: {gt_path.name}")
                continue
            if gt_path.name.startswith("."):
                logger.debug(f"Skipping hidden file: {gt_path.name}")
                continue
            if gt_path.suffix.lower() not in ImagePairUtil.SUPPORTED_EXTS:
                logger.warning(f"Unsupported file format: {gt_path.name}")
                continue

            stem = gt_path.stem.lower()
            pred_path = pred_index.get(stem)
            if not pred_path:
                logger.warning(f"No prediction found for: {gt_path.name}")
                continue

            if gt_path.suffix.lower() != pred_path.suffix.lower():
                pred_path = ImagePairUtil._convert_to_match(gt_path, pred_path)

            yield gt_path, pred_path

    @staticmethod
    def _convert_to_match(gt_path: Path, pred_path: Path):
        target_ext = gt_path.suffix.lower()
        orig_ext = pred_path.suffix.lower()

        new_path = pred_path.with_suffix(target_ext)
        backup_path = pred_path.with_name(f"{pred_path.stem}_orig{orig_ext}")

        try:
            img = Image.open(pred_path)

            if target_ext in (".jpg", ".jpeg"):
                img = img.convert("RGB")

            fmt = {
                ".png": "PNG",
                ".jpg": "JPEG",
                ".jpeg": "JPEG",
                ".webp": "WEBP",
            }.get(target_ext, "PNG")

            img.save(new_path, format=fmt)
            logger.info(f"Converted {pred_path.name} to {new_path.name}")

            if backup_path.exists():
                counter = 1
                while True:
                    alt = pred_path.with_name(f"{pred_path.stem}_orig_{counter}{orig_ext}")
                    if not alt.exists():
                        backup_path = alt
                        break
                    counter += 1
            pred_path.rename(backup_path)
            logger.info(f"Renamed original to {backup_path.name}")

            return new_path

        except Exception as e:
            logger.error(f"Failed to convert {pred_path.name}: {e}")
            return pred_path