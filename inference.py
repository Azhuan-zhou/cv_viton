import os
import math
import torch
import random
import argparse
import numpy as np
from PIL import Image
from transformers import CLIPVisionModelWithProjection

from src.pipeline_stable_diffusion_3_tryon import StableDiffusion3TryOnPipeline
from src.transformer_sd3_garm import SD3Transformer2DModel as SD3Transformer2DModel_Garm
from src.transformer_sd3_vton import SD3Transformer2DModel as SD3Transformer2DModel_Vton
from src.pose_guider import PoseGuider
from preprocess.dwpose import DWposeDetector
from preprocess.humanparsing.run_parsing import Parsing
from src.utils_mask import get_mask_location
'''example command:
python "./inference.py" \
  --model "./inputs/model.jpg" \
  --garment "./inputs/cloth.jpg" \
  --category "Upper-body" \
  --resolution 512x768\
  --steps 15 \
  --scale 3.0 \
'''

if torch.cuda.is_available():
    device = "cuda"
    weight_dtype = torch.float16
elif torch.backends.mps.is_available():
    device = "mps"
    weight_dtype = torch.float16
else:
    device = "cpu"
    weight_dtype = torch.float32


repo_path = "./local_model_dir"
output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)

print("Loading FitDiT modules...")
transformer_garm = SD3Transformer2DModel_Garm.from_pretrained(
    os.path.join(repo_path, "transformer_garm"), torch_dtype=weight_dtype
)
transformer_vton = SD3Transformer2DModel_Vton.from_pretrained(
    os.path.join(repo_path, "transformer_vton"), torch_dtype=weight_dtype
)
pose_guider = PoseGuider(
    conditioning_embedding_channels=1536,
    conditioning_channels=3,
    block_out_channels=(32, 64, 256, 512)
)
pose_guider.load_state_dict(torch.load(os.path.join(repo_path, "pose_guider", "diffusion_pytorch_model.bin")))
image_encoder_large = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=weight_dtype)
image_encoder_bigG = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", torch_dtype=weight_dtype)
pose_guider.to(device=device, dtype=weight_dtype)
image_encoder_large.to(device=device)
image_encoder_bigG.to(device=device)
pipeline = StableDiffusion3TryOnPipeline.from_pretrained(
    repo_path,
    torch_dtype=weight_dtype,
    transformer_garm=transformer_garm,
    transformer_vton=transformer_vton,
    pose_guider=pose_guider,
    image_encoder_large=image_encoder_large,
    image_encoder_bigG=image_encoder_bigG
)
pipeline.to(device)
dwpose = DWposeDetector(model_root=repo_path, device=device)
parsing_model = Parsing(model_root=repo_path, device=device)

print("All modules loaded successfully.")

def pad_and_resize(im, new_width=768, new_height=1024, pad_color=(255, 255, 255), mode=Image.LANCZOS):
    old_width, old_height = im.size
    ratio_w = new_width / old_width
    ratio_h = new_height / old_height

    if ratio_w < ratio_h:
        new_size = (new_width, round(old_height * ratio_w))
    else:
        new_size = (round(old_width * ratio_h), new_height)

    im_resized = im.resize(new_size, mode)
    pad_w = math.ceil((new_width - im_resized.width) / 2)
    pad_h = math.ceil((new_height - im_resized.height) / 2)

    new_im = Image.new('RGB', (new_width, new_height), pad_color)
    new_im.paste(im_resized, (pad_w, pad_h))

    return new_im, pad_w, pad_h


def unpad_and_resize(padded_im, pad_w, pad_h, original_width, original_height):
    width, height = padded_im.size
    left, top, right, bottom = pad_w, pad_h, width - pad_w, height - pad_h
    cropped_im = padded_im.crop((left, top, right, bottom))
    resized_im = cropped_im.resize((original_width, original_height), Image.LANCZOS)

    return resized_im

#no gradio
def generate_mask(vton_path, category="Upper-body", offsets=(0, 0, 0, 0)):
    with torch.inference_mode():
        vton_img = Image.open(vton_path)
        pose_img, keypoints, _, candidate = dwpose(np.array(vton_img)[:, :, ::-1])

        candidate[candidate < 0] = 0
        candidate = candidate[0]
        candidate[:, 0] *= vton_img.width
        candidate[:, 1] *= vton_img.height
        pose_img = Image.fromarray(pose_img[:, :, ::-1])

        model_parse, _ = parsing_model(vton_img)

        mask, mask_gray = get_mask_location(
            category, model_parse, candidate, model_parse.width, model_parse.height,
            *offsets
        )

        mask = mask.resize(vton_img.size).convert("L")
        #mask_gray = mask_gray.resize(vton_img.size).convert("L")

        return mask, pose_img


def run_inference(vton_path, garm_path, category="Upper-body",
                  resolution="768x1024", steps=20, scale=3.0, seed=-1):
    #assert resolution in ["768x1024", "1152x1536", "1536x2048"]
    new_width, new_height = map(int, resolution.split("x"))
    print(f"Running inference ({new_width}x{new_height}, steps={steps}, scale={scale})")
    mask, pose_img = generate_mask(vton_path, category=category)

    garm_img = Image.open(garm_path)
    vton_img = Image.open(vton_path)

    model_size = vton_img.size
    garm_img, _, _ = pad_and_resize(garm_img, new_width, new_height)
    vton_img, pad_w, pad_h = pad_and_resize(vton_img, new_width, new_height)
    mask, _, _ = pad_and_resize(mask, new_width, new_height)
    pose_img, _, _ = pad_and_resize(pose_img, new_width, new_height)

    if seed == -1:
        seed = random.randint(0, 2**31 - 1)

    with torch.inference_mode():
        outputs = pipeline(
            height=new_height,
            width=new_width,
            guidance_scale=scale,
            num_inference_steps=steps,
            generator=torch.Generator("cpu").manual_seed(seed),
            cloth_image=garm_img,
            model_image=vton_img,
            mask=mask,
            pose_image=pose_img,
            num_images_per_prompt=1
        ).images
    result = unpad_and_resize(outputs[0], pad_w, pad_h, model_size[0], model_size[1])
    return result


# 命令行接口
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FitDiT Local Inference v2")
    parser.add_argument("--model", type=str, required=True, help="Path to model image (person)")
    parser.add_argument("--garment", type=str, required=True, help="Path to garment image")
    parser.add_argument("--category", type=str, default="Upper-body", help="Garment category")
    parser.add_argument("--resolution", type=str, default="768x1024", help="Try-on resolution")
    parser.add_argument("--steps", type=int, default=20, help="Diffusion steps")
    parser.add_argument("--scale", type=float, default=3.0, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")
    parser.add_argument("--save_name", type=str, default=None, help="Output filename")

    args = parser.parse_args()

    result_img = run_inference(
        vton_path=args.model,
        garm_path=args.garment,
        category=args.category,
        resolution=args.resolution,
        steps=args.steps,
        scale=args.scale,
        seed=args.seed
    )
    if args.save_name is None:
        model_basename = os.path.basename(args.model)
        args.save_name = model_basename
    save_path = os.path.join(output_dir, args.save_name)
    result_img.save(save_path)
    print(f"[SUCCESS] Output saved at: {save_path}")
