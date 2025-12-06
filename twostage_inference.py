import os
import torch
import argparse
from inference import run_inference, generate_mask
from diffusers import StableDiffusionXLInpaintPipeline

def load_sdxl_inpaint(device=None):

    print("Loading SDXL Inpainting model for prompt editing...")
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if device in ["cpu", "mps"]:
        dtype = torch.float32
    else:
        dtype = torch.float16

    pipe=StableDiffusionXLInpaintPipeline.from_pretrained(
            "./local_model_dir/sdxl_inpaint",
            torch_dtype=dtype,
    ).to(device)

    return pipe, device

def apply_prompt_editing(pipe, img, mask, prompt):
    print(f"\n[Stage 2] Prompt editing with prompt: {prompt}")

    mask_rgb = mask.convert("RGB")

    edited = pipe(
        prompt=prompt,
        image=img,
        mask_image=mask_rgb,
        strength=0.85,
        height=img.height,
        width = img.width
    ).images[0]

    return edited

def main(args):

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n[Stage 1] Running FitDiT Try-on...")

    tryon_img = run_inference(
        vton_path=args.model,
        garm_path=args.garment,
        category=args.category,
        resolution=args.resolution,
        steps=args.steps,
        scale=args.scale,
        seed=args.seed
    )

    stage1_path = os.path.join(args.output_dir, "tryon_result.jpg")
    tryon_img.save(stage1_path, "JPEG")

    mask, _ = generate_mask(args.model, category=args.category)
    mask = mask.resize(tryon_img.size)

    mask_path = os.path.join(args.output_dir, "cloth_mask.jpg")
    mask.save(mask_path, "JPEG")

    print("\nDo you want to edit the clothing with a prompt?")
    choice = input("Enter 1 or 0: ").strip()

    if choice == "0":
        print("No editing applied. Done.")
        return

    if choice != "1":
        print("Invalid input. Exit.")
        return

    prompt = input("\nEnter your prompt (e.g., 'make the shirt blue'): ").strip()

    print("\nLoading Inpainting model...")
    pipe, dev = load_sdxl_inpaint()

    print(f"SDXL loaded on device: {dev}")

    edited_img = apply_prompt_editing(pipe, tryon_img, mask, prompt)

    final_path = os.path.join(args.output_dir, "prompt_tryon.jpg")
    edited_img.save(final_path, "JPEG")

    print(f"\n[SUCCESS] Final edited image saved at: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FitDiT Local Inference v3 two stage prompt")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--garment", type=str, required=True)
    parser.add_argument("--category", type=str, default="Upper-body")
    parser.add_argument("--resolution", type=str, default="768x1024")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default="./twostage_output")

    args = parser.parse_args()
    main(args)
