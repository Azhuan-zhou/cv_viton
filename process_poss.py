

from preprocess.dwpose import DWposeDetector
from tqdm import tqdm

import torch
import PIL.Image as Image
import numpy as np
from src.utils_mask import get_mask_location
from preprocess.humanparsing.run_parsing import Parsing
parsing_model = Parsing(model_root=repo_path, device=device)
repo_path = "./local_model_dir"

device = "cuda"
dwprocessor = DWposeDetector(model_root=repo_path, device=device)
def generate_mask(vton_img, category="dress", offset_top=0, offset_bottom=0, offset_left=0, offset_right=0):
    name = vton_img.split('/')[-1].split('.')[0]
    with torch.inference_mode():
        vton_img = Image.open(vton_img)

        pose_image, keypoints, _, candidate = dwprocessor(np.array(vton_img)[:,:,::-1])
        candidate[candidate<0]=0
        candidate = candidate[0]

        candidate[:, 0]*=vton_img.width
        candidate[:, 1]*=vton_img.height

        pose_image = pose_image[:,:,::-1] #rgb
        pose_image = Image.fromarray(pose_image)
        model_parse, _ = parsing_model(vton_img)

        
        mask, mask_gray = get_mask_location(category, model_parse, \
                                    candidate, model_parse.width, model_parse.height, \
                                    offset_top, offset_bottom, offset_left, offset_right)
        mask = mask.resize(vton_img.size)
        mask_gray = mask_gray.resize(vton_img.size)
        mask = mask.convert("L")

    
    #pose_image.save(f"./viton-hd/test/image-densepose-new/{name}.jpg")
    mask.save(f"./viton-hd/test/mask-new/{name}.png")
    return pose_image

if __name__ == "__main__":
    import os
    data_root = "./viton-hd/test/image"
    images = [os.path.join(data_root, i ) for i in os.listdir(data_root)]
    for img in tqdm(images, desc="Generating densepose maps", total=len(images)):
        generate_mask(img)