"""
Luna USDU Clone - Exact replication of Ultimate SD Upscale logic

This is a surgical extraction of USDU's core tiling logic for testing and experimentation.
Maintains the exact flow: crop pixels → encode → sample → decode → paste per tile.

Purpose: Understand why USDU works and our chess refiner doesn't.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import comfy.sample
import comfy.utils
from nodes import VAEEncode, VAEDecode


def pil_to_tensor(image):
    """Convert PIL to ComfyUI tensor format [B, H, W, C] - copied from USDU"""
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0)
    if len(image.shape) == 3:  # If grayscale, add channel dimension
        image = image.unsqueeze(-1)
    return image


def tensor_to_pil(img_tensor, batch_index=0):
    """Convert ComfyUI tensor to PIL - copied from USDU"""
    safe_tensor = torch.nan_to_num(img_tensor[batch_index])
    return Image.fromarray((255 * safe_tensor.cpu().numpy()).astype(np.uint8))


class LunaUSDUClone:
    """
    Faithful clone of Ultimate SD Upscale's tiling logic.
    Sequential processing, per-tile encode/decode.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.5}),
                "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "tile_size": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Tile size for refinement"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale_tiled"
    CATEGORY = "Luna/Detailing/Debug"
    
    def upscale_tiled(
        self,
        image: torch.Tensor,
        model,
        vae,
        positive,
        negative,
        steps: int,
        cfg: float,
        denoise: float,
        seed: int,
        sampler_name: str,
        scheduler: str,
        tile_size: int
    ):
        """
        Clone of USDU's linear tiling process.
        
        Exact flow per tile:
        1. Crop pixels
        2. Encode → latent
        3. Sample
        4. Decode → pixels
        5. Paste back to canvas
        """
        device = comfy.model_management.get_torch_device()
        
        # Convert to PIL (USDU does this)
        b, h, w, c = image.shape
        pil_image = tensor_to_pil(image, 0)
        
        # Calculate grid
        cols = int(np.ceil(w / tile_size))
        rows = int(np.ceil(h / tile_size))
        total_tiles = rows * cols
        
        print(f"[LunaUSDUClone] Processing {w}×{h} image")
        print(f"[LunaUSDUClone] Grid: {rows}×{cols} = {total_tiles} tiles")
        print(f"[LunaUSDUClone] Tile size: {tile_size}×{tile_size}")
        
        # Create output canvas
        output_canvas = pil_image.copy()
        
        # Create VAE encoder/decoder instances (like USDU does)
        vae_encoder = VAEEncode()
        vae_decoder = VAEDecode()
        
        # Process tiles sequentially (USDU style)
        tile_count = 0
        for yi in range(rows):
            for xi in range(cols):
                tile_count += 1
                
                # Calculate tile region
                x1 = xi * tile_size
                y1 = yi * tile_size
                x2 = min(x1 + tile_size, w)
                y2 = min(y1 + tile_size, h)
                
                print(f"[LunaUSDUClone] Tile {tile_count}/{total_tiles}: ({x1},{y1}) to ({x2},{y2})")
                
                # STEP 1: Crop pixels (USDU does this)
                tile_pil = pil_image.crop((x1, y1, x2, y2))
                actual_w, actual_h = tile_pil.size
                
                # Resize if not exact tile size (edge tiles)
                if actual_w != tile_size or actual_h != tile_size:
                    tile_pil = tile_pil.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
                
                # STEP 2: Encode pixels → latent (USDU does this)
                # Convert PIL to tensor
                tile_tensor = pil_to_tensor(tile_pil)
                # Use VAEEncode node exactly like USDU
                (tile_latent,) = vae_encoder.encode(vae, tile_tensor)
                
                # STEP 3: Sample (img2img refinement)
                # USDU uses global conditioning (no cropping for plain CLIP)
                refined_latent = comfy.sample.sample(
                    model,
                    noise=torch.randn_like(tile_latent["samples"]),
                    steps=steps,
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    positive=positive,
                    negative=negative,
                    latent_image=tile_latent["samples"],
                    denoise=denoise,
                    disable_noise=False,
                    start_step=None,
                    last_step=None,
                    force_full_denoise=True,
                    noise_mask=None,
                    sigmas=None,
                    callback=None,
                    disable_pbar=False,
                    seed=seed + tile_count
                )
                
                # STEP 4: Decode → pixels (use VAEDecode node like USDU)
                refined_latent_dict = {"samples": refined_latent}
                (decoded_tensor,) = vae_decoder.decode(vae, refined_latent_dict)
                
                # Convert back to PIL
                decoded_pil = tensor_to_pil(decoded_tensor, 0)
                
                # Resize back if needed
                if actual_w != tile_size or actual_h != tile_size:
                    decoded_pil = decoded_pil.resize((actual_w, actual_h), Image.Resampling.LANCZOS)
                
                # STEP 5: Paste back to canvas (simple paste like USDU)
                output_canvas.paste(decoded_pil, (x1, y1))
        
        print(f"[LunaUSDUClone] Complete!")
        
        # Convert back to tensor
        output_tensor = pil_to_tensor(output_canvas)
        
        return (output_tensor,)


NODE_CLASS_MAPPINGS = {
    "LunaUSDUClone": LunaUSDUClone,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaUSDUClone": "Luna USDU Clone (Debug)",
}
