# Copyright 2024 Stability AI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.training_utils import compute_snr
import torch
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, SD3LoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
from transformers import CLIPImageProcessor
import torchvision.transforms as transforms
import torch.nn.functional as F
from src.pose_guider import PoseGuider
from transformers import CLIPVisionModelWithProjection

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusion3Pipeline

        >>> pipe = StableDiffusion3Pipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
        ... )
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> image = pipe(prompt).images[0]
        >>> image.save("sd3.png")
        ```
"""

def _fft_mag(x):
    if x.shape[1] > 1:
        x = x.mean(dim=1, keepdim=True)
    X = torch.fft.fft2(x)            
    mag = torch.abs(X)
    return mag

def fft_loss_function(x, y):
    mag_x = _fft_mag(x)
    mag_x = torch.log1p(mag_x)
    mag_y = _fft_mag(y)
    mag_y = torch.log1p(mag_y)
    loss = F.mse_loss(mag_x, mag_y)
    return loss

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class StableDiffusion3TryOnPipeline(DiffusionPipeline, SD3LoraLoaderMixin, FromSingleFileMixin):

    model_cpu_offload_seq = "image_encoder_large->image_encoder_bigG->pose_guider->transformer_garm->transformer_vton->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "negative_pooled_prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        transformer_garm: SD3Transformer2DModel,
        transformer_vton: SD3Transformer2DModel,
        pose_guider: PoseGuider,
        image_encoder_large: CLIPVisionModelWithProjection,
        image_encoder_bigG: CLIPVisionModelWithProjection,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            transformer_garm=transformer_garm,
            transformer_vton=transformer_vton,
            scheduler=scheduler,
            pose_guider=pose_guider,
            image_encoder_large=image_encoder_large,
            image_encoder_bigG=image_encoder_bigG,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )

        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 128
        )
        self.vit_processing = CLIPImageProcessor()
        
        self.transformer_vton.requires_grad_(True)
        self.transformer_garm.requires_grad_(False)
        self.image_encoder_large.requires_grad_(False)
        self.image_encoder_bigG.requires_grad_(False)
        self.pose_guider.requires_grad_(False)
        self.vae.requires_grad_(False)
        
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.text_encoder_3 is None:
            return torch.zeros(
                (
                    batch_size * num_images_per_prompt,
                    self.tokenizer_max_length,
                    self.transformer.config.joint_attention_dim,
                ),
                device=device,
                dtype=dtype,
            )

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_3.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_3(text_input_ids.to(device))[0]

        dtype = self.text_encoder_3.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        clip_skip: Optional[int] = None,
        clip_model_index: int = 0,
    ):
        device = device or self._execution_device

        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        prompt_3: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        clip_skip: Optional[int] = None,
        max_sequence_length: int = 256,
        lora_scale: Optional[float] = None,
    ):
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, SD3LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            prompt_3 = prompt_3 or prompt
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

            prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=0,
            )
            prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                prompt=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=1,
            )
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

            t5_prompt_embed = self._get_t5_prompt_embeds(
                prompt=prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
            )

            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt_3 = negative_prompt_3 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )
            negative_prompt_3 = (
                batch_size * [negative_prompt_3] if isinstance(negative_prompt_3, str) else negative_prompt_3
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embed, negative_pooled_prompt_embed = self._get_clip_prompt_embeds(
                negative_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=0,
            )
            negative_prompt_2_embed, negative_pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                negative_prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=1,
            )
            negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)

            t5_negative_prompt_embed = self._get_t5_prompt_embeds(
                prompt=negative_prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            negative_clip_prompt_embeds = torch.nn.functional.pad(
                negative_clip_prompt_embeds,
                (0, t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1]),
            )

            negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
            )

        if self.text_encoder is not None:
            if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def check_inputs(
        self,
        height,
        width,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )


    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def _get_clip_image_embeds(self, cloth, num_images_per_prompt, device):
        image_embeds_large = self.image_encoder_large(cloth).image_embeds
        image_embeds_bigG = self.image_encoder_bigG(cloth).image_embeds
        return torch.cat([image_embeds_large, image_embeds_bigG], dim=1)

    def prepare_image_latents(
            self,
            image
    ):
        image_latents = self.vae.encode(image).latent_dist.sample()
        image_latents = (image_latents-self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return image_latents



    def forward(self,batch,
                num_images_per_prompt=1,
                fft_scale=0.1,
                callback_on_step_end_tensor_inputs: List[str] = ["latents"],
                joint_attention_kwargs: Optional[Dict[str, Any]] = None,
                clip_skip: Optional[int] = None,
                generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                snr_gamma: Optional[float] = None,):
        device = self._execution_device
        pixel_values = batch["image"]
        B, _, H, W = pixel_values.shape
        cloth_image = batch["cloth_pure"]
        viton_image = batch["im_mask"]
        mask = batch["inpaint_mask"]
        pose_image = batch["pose_img"]


        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            H,
            W,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters

        cloth_image_vit = batch["cloth"]
        cloth_image_vit = cloth_image_vit.to(device=device)
        cloth_image_enbeds = self._get_clip_image_embeds(cloth_image_vit, num_images_per_prompt, device)
        cloth_image_enbeds = cloth_image_enbeds.to(device=device)
        cloth_image_enbeds = cloth_image_enbeds.repeat(num_images_per_prompt, *([1] * (cloth_image_enbeds.dim() - 1)))
        
        # 3. vae latent
        viton_image = viton_image.to(device)
        model_input = pixel_values.to(device)
        cloth_image = cloth_image.to(device)
        pose_image = pose_image.to(device)
        mask = mask.to(device)
        mask = F.interpolate(mask, (mask.shape[2]//8, mask.shape[3]//8))
        
        pose_fea = self.pose_guider(pose_image.to(device=device, dtype=cloth_image_enbeds.dtype))
        vton_model_latents = self.prepare_image_latents(viton_image)
        garm_model_latents = self.prepare_image_latents(cloth_image)
        input_model_latents = self.prepare_image_latents(model_input)

        
        # 4. Prepare timesteps
        idx = torch.randint(0, self.scheduler.timesteps.shape[0], (B,))
        t = self.scheduler.timesteps[idx].to(device=device, dtype=input_model_latents.dtype)
        noise = torch.randn_like(input_model_latents).to(device=device)

        latents = self.scheduler.scale_noise(input_model_latents,t,noise)
        
    
        
        # 5. Prepare latent model inputs

        _, ref_key, ref_value = self.transformer_garm(hidden_states=garm_model_latents,
                                        timestep=t * 0,
                                        pooled_projections=cloth_image_enbeds,
                                        encoder_hidden_states=None,
                                        return_dict=False)
        print(latents.shape, vton_model_latents.shape, mask.shape)
        noise_pred = self.transformer_vton(hidden_states=torch.cat([latents, vton_model_latents, mask], dim=1),
                            timestep=t,
                            pooled_projections=cloth_image_enbeds,
                            encoder_hidden_states=None,
                            ref_key=ref_key,
                            ref_value=ref_value,
                            return_dict=False,
                            pose_cond=pose_fea)[0]
        

        denoise_loss = F.mse_loss(noise_pred, noise, reduction="mean")

        num_steps = float(self.scheduler.config.num_train_timesteps)
        t_cont = (t.float()+0.5)/ num_steps
        t_cont = t_cont.view(-1, 1, 1, 1) 
        z0_hat = (latents - t_cont * noise_pred) / (1.0 - t_cont) 
        z0_hat = ( z0_hat / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image_pred = self.vae.decode( z0_hat, return_dict=False)[0]
        viton_mask = batch['mask'].to(device)
        viton_pred =  image_pred * (viton_mask > 0.5)
        viton_true = model_input * (viton_mask > 0.5)

        F_tr = torch.fft.fftn(viton_pred, dim=(-2, -1))
        F_p  = torch.fft.fftn(viton_true,  dim=(-2, -1))
        F_diff = F_tr - F_p
        F_dist_sq = F_diff.real**2 + F_diff.imag**2
        fft_loss = F_dist_sq.mean()
        total_loss = denoise_loss + fft_scale * fft_loss
        
        print(f"denoise_loss: {denoise_loss.item()}, fft_loss: {fft_loss.item()}")
        return total_loss, denoise_loss, fft_loss
    
    @torch.no_grad()
    def inference(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        cloth_image=None,
        model_image=None,
        mask=None,
        pose_image=None
    ):

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            height,
            width,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        batch_size = cloth_image.shape[0]

        device = self._execution_device



        cloth_image_vit = cloth_image.to(device=device)
        cloth_image_enbeds = self._get_clip_image_embeds(cloth_image_vit, num_images_per_prompt, device)
        cloth_image_enbeds = cloth_image_enbeds.to(device=device)
        cloth_image_enbeds = cloth_image_enbeds.repeat(num_images_per_prompt, *([1] * (cloth_image_enbeds.dim() - 1)))
        
        if self.do_classifier_free_guidance:
            cloth_image_enbeds = torch.cat([torch.zeros_like(cloth_image_enbeds), cloth_image_enbeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer_garm.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            cloth_image_enbeds.dtype,
            device,
            generator,
            latents,
        )



        model_image = model_image.to(latents)
        cloth_image = cloth_image.to(latents)
        mask = mask.to(latents)
        vton_image = model_image * (mask < 0.5)
        mask = F.interpolate(mask, (vton_image.shape[2]//8, vton_image.shape[3]//8))
        mask = mask[:, 0:1]
        pose_fea = self.pose_guider(pose_image.to(device=latents.device, dtype=latents.dtype))
        vton_model_latents = self.prepare_image_latents(vton_image)
        garm_model_latents = self.prepare_image_latents(cloth_image)
       
        
        
        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                vton_model_input = torch.cat([vton_model_latents] * 2) if self.do_classifier_free_guidance else vton_model_latents
                mask_input = torch.cat([mask] * 2) if self.do_classifier_free_guidance else mask
                garm_model_input = torch.cat([torch.zeros_like(garm_model_latents), garm_model_latents]) if self.do_classifier_free_guidance else garm_model_latents
                pose_fea_input = torch.cat([pose_fea] * 2) if self.do_classifier_free_guidance else pose_fea
                
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])
                if i==0:
                    _, ref_key, ref_value = self.transformer_garm(hidden_states=garm_model_input,
                                        timestep=timestep * 0,
                                        pooled_projections=cloth_image_enbeds,
                                        encoder_hidden_states=None,
                                        return_dict=False)
                noise_pred = self.transformer_vton(hidden_states=torch.cat([latent_model_input, vton_model_input, mask_input], dim=1),
                            timestep=timestep,
                            pooled_projections=cloth_image_enbeds,
                            encoder_hidden_states=None,
                            ref_key=ref_key,
                            ref_value=ref_value,
                            return_dict=False,
                            pose_cond=pose_fea_input)[0]
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)


                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)
        
        
