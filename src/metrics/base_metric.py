from abc import ABC, abstractmethod
from PIL import Image

class MetricService(ABC):
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def compute(self, gt_img: Image.Image, pred_img: Image.Image):
        pass