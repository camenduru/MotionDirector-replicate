from cog import BasePredictor, Input, Path
import sys
sys.path.append('/content/MotionDirector')

import argparse
import os
import platform
import re
import warnings
from typing import Optional

import torch
from diffusers import DDIMScheduler, TextToVideoSDPipeline
from einops import rearrange
from torch import Tensor
from torch.nn.functional import interpolate
from tqdm import trange
import random

from MotionDirector_train import export_to_video, handle_memory_attention, load_primary_models, unet_and_text_g_c, freeze_models
from utils.lora_handler import LoraHandler
from utils.ddim_utils import ddim_inversion
import imageio

def initialize_pipeline(
    model: str,
    device: str = "cuda",
    xformers: bool = False,
    sdp: bool = False,
    lora_path: str = "",
    lora_rank: int = 64,
    lora_scale: float = 1.0,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scheduler, tokenizer, text_encoder, vae, unet = load_primary_models(model)
    # Freeze any necessary models
    freeze_models([vae, text_encoder, unet])
    # Enable xformers if available
    handle_memory_attention(xformers, sdp, unet)
    lora_manager_temporal = LoraHandler(
        version="cloneofsimo",
        use_unet_lora=True,
        use_text_lora=False,
        save_for_webui=False,
        only_for_webui=False,
        unet_replace_modules=["TransformerTemporalModel"],
        text_encoder_replace_modules=None,
        lora_bias=None
    )
    unet_lora_params, unet_negation = lora_manager_temporal.add_lora_to_model(
        True, unet, lora_manager_temporal.unet_replace_modules, 0, lora_path, r=lora_rank, scale=lora_scale)
    unet.eval()
    text_encoder.eval()
    unet_and_text_g_c(unet, text_encoder, False, False)
    pipe = TextToVideoSDPipeline.from_pretrained(
        pretrained_model_name_or_path=model,
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder.to(device=device, dtype=torch.half),
        vae=vae.to(device=device, dtype=torch.half),
        unet=unet.to(device=device, dtype=torch.half),
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe

def inverse_video(pipe, latents, num_steps):
    ddim_inv_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    ddim_inv_scheduler.set_timesteps(num_steps)
    ddim_inv_latent = ddim_inversion(
        pipe, ddim_inv_scheduler, video_latent=latents.to(pipe.device),
        num_inv_steps=num_steps, prompt="")[-1]
    return ddim_inv_latent

def prepare_input_latents(
    pipe: TextToVideoSDPipeline,
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    latents_path:str,
    noise_prior: float
):
    # initialize with random gaussian noise
    scale = pipe.vae_scale_factor
    shape = (batch_size, pipe.unet.config.in_channels, num_frames, height // scale, width // scale)
    if noise_prior > 0.:
        cached_latents = torch.load(latents_path)
        if 'inversion_noise' not in cached_latents:
            latents = inverse_video(pipe, cached_latents['latents'].unsqueeze(0), 50).squeeze(0)
        else:
            latents = torch.load(latents_path)['inversion_noise'].unsqueeze(0)
        if latents.shape[0] != batch_size:
            latents = latents.repeat(batch_size, 1, 1, 1, 1)
        if latents.shape != shape:
            latents = interpolate(rearrange(latents, "b c f h w -> (b f) c h w", b=batch_size), (height // scale, width // scale), mode='bilinear')
            latents = rearrange(latents, "(b f) c h w -> b c f h w", b=batch_size)
        noise = torch.randn_like(latents, dtype=torch.half)
        latents = (noise_prior) ** 0.5 * latents + (1 - noise_prior) ** 0.5 * noise
    else:
        latents = torch.randn(shape, dtype=torch.half)
    return latents

def encode(pipe: TextToVideoSDPipeline, pixels: Tensor, batch_size: int = 8):
    nf = pixels.shape[2]
    pixels = rearrange(pixels, "b c f h w -> (b f) c h w")
    latents = []
    for idx in trange(
        0, pixels.shape[0], batch_size, desc="Encoding to latents...", unit_scale=batch_size, unit="frame"
    ):
        pixels_batch = pixels[idx : idx + batch_size].to(pipe.device, dtype=torch.half)
        latents_batch = pipe.vae.encode(pixels_batch).latent_dist.sample()
        latents_batch = latents_batch.mul(pipe.vae.config.scaling_factor).cpu()
        latents.append(latents_batch)
    latents = torch.cat(latents)
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=nf)
    return latents

@torch.inference_mode()
def inference(
    pipe,
    prompt: str,
    negative_prompt: Optional[str] = None,
    width: int = 256,
    height: int = 256,
    num_frames: int = 24,
    num_steps: int = 50,
    guidance_scale: float = 15,
    device: str = "cuda",
    xformers: bool = False,
    sdp: bool = False,
    lora_path: str = "",
    lora_rank: int = 64,
    lora_scale: float = 1.0,
    seed: Optional[int] = None,
    latents_path: str="",
    noise_prior: float = 0.,
    repeat_num: int = 1,
    output_dir: str = "/content",
    fps: int = 256,
):
    if seed is not None:
        random_seed = seed
        torch.manual_seed(seed)

    with torch.autocast(device, dtype=torch.half):
        # prepare models
        # pipe = initialize_pipeline(model, device, xformers, sdp, lora_path, lora_rank, lora_scale)
        for i in range(repeat_num):
            if seed is None:
                random_seed = random.randint(100, 10000000)
                torch.manual_seed(random_seed)
            # prepare input latents
            init_latents = prepare_input_latents(
                pipe=pipe,
                batch_size=len(prompt),
                num_frames=num_frames,
                height=height,
                width=width,
                latents_path=latents_path,
                noise_prior=noise_prior
            )

            with torch.no_grad():
                video_frames = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    latents=init_latents
                ).frames

            os.makedirs(output_dir, exist_ok=True)

            # save to mp4
            # export_to_video(video_frames, f"{out_name}_{random_seed}.mp4", fps)

            # # save to gif
            # file_name = f"{out_name}_{random_seed}.gif"
            # imageio.mimsave(file_name, video_frames, 'GIF', duration=1000 * 1 / fps, loop=0)

    return video_frames

class Predictor(BasePredictor):
    def setup(self) -> None:
        model = "/content/MotionDirector/models/MotionDirector/zeroscope_v2_576w"
        self.device = "cuda"
        self.xformers = True
        self.sdp = False
        self.checkpoint_folder = "/content/MotionDirector/models/MotionDirector/train/train_2023-12-02T13-39-36"
        checkpoint_index = 300
        self.lora_path = f"{self.checkpoint_folder}/checkpoint-{checkpoint_index}/temporal/lora"
        self.lora_rank = 32
        self.lora_scale = 1.0
        with torch.autocast(self.device, dtype=torch.half):
            self.pipe = initialize_pipeline(model, self.device, self.xformers, self.sdp, self.lora_path, self.lora_rank, self.lora_scale)
    def predict(
        self,
        prompt: str = Input(default="A person is riding a bicycle past the Eiffel Tower."),
        negative_prompt: str = Input(default="blurry"),
        width: int = Input(default=256, ge=256, le=512),
        height: int = Input(default=256, ge=256, le=512),
        num_frames: int = Input(default=24, ge=0, le=48),
        num_steps: int = Input(default=50, ge=1, le=150),
        guidance_scale: int = Input(default=15, ge=1, le=25),
        seed: int = Input(default=15, ge=0, le=1000000),
    ) -> Path:
        prompt = f"\"{prompt}\""
        negative_prompt = f"\"{negative_prompt}\""
        noise_prior = 0
        repeat_num = 1
        batch_size = 1
        output_dir = f"/tmp"
        out_name = f"{output_dir}/"
        prompt = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", prompt) if platform.system() == "Windows" else prompt
        out_name += f"{prompt}".replace(' ','_').replace(',', '').replace('.', '')
        prompt = [prompt] * batch_size
        if negative_prompt is not None:
            negative_prompt = [negative_prompt] * batch_size
        latents_folder = f"{self.checkpoint_folder}/cached_latents"
        latents_path = f"{latents_folder}/{random.choice(os.listdir(latents_folder))}"
        assert os.path.exists(self.lora_path)
        video_frames = inference(
            pipe=self.pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            device=self.device,
            xformers=self.xformers,
            sdp=self.sdp,
            lora_path=self.lora_path,
            lora_rank=self.lora_rank,
            lora_scale = self.lora_scale,
            seed=seed,
            latents_path=latents_path,
            noise_prior=noise_prior,
            repeat_num=repeat_num
        )
        output = f"{out_name}_{seed}.mp4"
        export_to_video(video_frames, output, 8)
        return Path(output)