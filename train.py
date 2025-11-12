import torch
from VitonDataset import VitonHDDataset
from src.pipeline_tryon_train import StableDiffusion3TryOnPipeline
import spaces

import os
import argparse
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
import torch
import torch.nn as nn
import numpy as np
from src.transformer_sd3_garm import SD3Transformer2DModel as SD3Transformer2DModel_Garm
from src.transformer_sd3_vton import SD3Transformer2DModel as SD3Transformer2DModel_Vton
from src.pose_guider import PoseGuider
import torch.nn.functional as F
from diffusers.training_utils import compute_snr
import itertools
weight_dtype = torch.float32


def reinit_module(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)

    elif hasattr(m, "reset_parameters"):
        try:
            m.reset_parameters()
        except:
            pass

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--data_dir", type=str, default="./viton-hd", help="path to dataset")
    parser.add_argument("--output_dir", type=str, default="./samples", help="path to save the model")
    parser.add_argument("--width",type=int,default= 384,)
    parser.add_argument("--height",type=int,default=512,)
    parser.add_argument("--repo_path", type=str, default="./local_model_dir", help="path to local model repo")
    parser.add_argument("--device", type=str, default="cuda", help="device to use for training")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--test_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--epochs", type=int, default=100, help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate",type=float,default=1e-5,help="Learning rate to use.",)
    parser.add_argument("--snr_gamma",type=float,default=None,help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. ""More details here: https://arxiv.org/abs/2303.09556.",)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    device = args.device
 
    transformer_garm = SD3Transformer2DModel_Garm.from_pretrained(os.path.join(args.repo_path, "transformer_garm"), torch_dtype=weight_dtype,local_files_only=True)
    transformer_vton = SD3Transformer2DModel_Vton.from_pretrained(os.path.join(args.repo_path, "transformer_vton"), torch_dtype=weight_dtype,local_files_only=True)
    transformer_vton.apply(reinit_module)
    pose_guider =  PoseGuider(conditioning_embedding_channels=1536, conditioning_channels=3, block_out_channels=(32, 64, 256, 512))
    pose_guider.load_state_dict(torch.load(os.path.join(args.repo_path, "pose_guider", "diffusion_pytorch_model.bin")))
    image_encoder_large = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=weight_dtype)
    image_encoder_bigG = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", torch_dtype=weight_dtype)
    pose_guider.to(device=device, dtype=weight_dtype)
    image_encoder_large.to(device=device)
    image_encoder_bigG.to(device=device)

    pipeline = StableDiffusion3TryOnPipeline.from_pretrained(args.repo_path, torch_dtype=weight_dtype, \
            transformer_garm=transformer_garm, transformer_vton=transformer_vton, pose_guider=pose_guider, \
            image_encoder_large=image_encoder_large, image_encoder_bigG=image_encoder_bigG,local_files_only=True)

    pipeline.to(device=device)
    
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = itertools.chain(pipeline.transformer_vton.parameters())
    optimizer = optimizer_class( params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    train_dataset = VitonHDDataset(
        dataroot_path=args.data_dir,
        phase="train",
        order="paired",
        size=(args.height, args.width),
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        pin_memory=True,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=16,
    )
    
    test_dataset = VitonHDDataset(
        dataroot_path=args.data_dir,
        phase="test",
        order="paired",
        size=(args.height, args.width),
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        num_workers=4,
    )
    
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_loss_denoise = 0.0
        total_loss_fft = 0.0
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            loss, denoise_loss, fft_loss = pipeline.forward(batch)

            total_loss += loss.item()
            total_loss_denoise += denoise_loss.item()
            total_loss_fft += fft_loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % 1000 == 0:
                total_loss /= 1000
                total_loss_denoise /= 1000
                total_loss_fft /= 1000
                print(f"Epoch {epoch}, Step {step}, Loss: {total_loss:.4f}, Denoise Loss: {total_loss_denoise:.4f}, FFT Loss: {total_loss_fft:.4f}")
                total_loss = 0.0
                total_loss_denoise = 0.0
                total_loss_fft = 0.0
            
        if epoch % 10 == 0:
            with torch.no_grad():

                images = []
                total_loss = 0.0
                total_loss_denoise = 0.0
                total_loss_fft = 0.0
                for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                    if step == 0:
                        images = pipeline.inference(
                            batch=batch,
                            num_inference_steps=20,
                        ).images
                        for i in range(len(images)):
                            images[i].save(os.path.join(args.output_dir,str(epoch)+"_"+str(i)+"_"+"test.jpg"))
                    loss, denoise_loss, fft_loss = pipeline.forward(batch=batch)
                    total_loss += loss.item()
                    total_loss_denoise += denoise_loss.item()
                    total_loss_fft += fft_loss.item()
                avg_loss = total_loss / (step + 1)
                avg_loss_denoise = total_loss_denoise / (step + 1)
                avg_loss_fft = total_loss_fft / (step + 1)
                print(f"Epoch {epoch}, Validation Total Loss: {avg_loss:.4f}, Denoise Loss: {avg_loss_denoise:.4f}, FFT Loss: {avg_loss_fft:.4f}")
            
            pipeline.transformer_vton.save_pretrained("ckpt_dir/epoch_{}".format(epoch), safe_serialization=True)



if __name__ == "__main__":
    main()