import torch
import numpy as np
import node_helpers
import sys
import os

# Add the parent directory to sys.path to enable relative imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utils modules directly
utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils")
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

try:
    from mediapipe_engine import Mediapipe_Engine  # type: ignore
except ImportError:
    print("Luna MediaPipe Detailer: mediapipe_engine not available")
    Mediapipe_Engine = None

try:
    from utils.luna_performance_monitor import LunaPerformanceMonitor
except ImportError:
    print("Luna MediaPipe Detailer: Performance monitoring not available")
    LunaPerformanceMonitor = None

ENGINE_INSTANCE = None

class Luna_MediaPipe_Detailer:
    @classmethod
    def INPUT_TYPES(cls):
        model_types = ["hands", "face", "eyes", "mouth", "feet", "torso", "full_body (bbox)", "person (segmentation)"]
        sort_by_types = ["Confidence", "Area: Largest to Smallest", "Position: Left to Right", "Position: Top to Bottom", "Position: Nearest to Center"]
        return {
            "required": {
                "image": ("IMAGE",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "model_type": (model_types,),
                "sort_by": (sort_by_types,),
                "max_objects": ("INT", {"default": 10, "min": 1, "max": 10, "step": 1}),
                "confidence": ("FLOAT", {"default": 0.30, "min": 0.1, "max": 1.0, "step": 0.05}),
                "mask_padding": ("INT", {"default": 35, "min": 0, "max": 200, "step": 1}),
                "mask_blur": ("INT", {"default": 6, "min": 0, "max": 100, "step": 1}),
                "noise_mask": ("BOOLEAN", {"default": True}),
                "detail_positive_prompt": ("STRING", {"multiline": True, "default": "A high-resolution photograph with realistic details, high quality"}),
                "detail_negative_prompt": ("STRING", {"multiline": True, "default": "blurry, low quality, deformed"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "flux_guidance_strength": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 30.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "FLOAT", "MASK", "IMAGE", "CLIP", "VAE", "IMAGE", "PERFORMANCE_STATS")
    RETURN_NAMES = ("POSITIVE", "NEGATIVE", "LATENT", "DENOISE", "MASK", "IMAGE_PASSTHROUGH", "CLIP_PASSTHROUGH", "VAE_PASSTHROUGH", "MASKED_PREVIEW", "performance_stats")
    FUNCTION = "process"
    CATEGORY = "Luna/Detailing"

    def process(self, image, clip, vae, model_type, sort_by, max_objects, confidence, mask_padding, mask_blur, noise_mask, detail_positive_prompt, detail_negative_prompt, denoise, flux_guidance_strength):
        global ENGINE_INSTANCE

        # Initialize performance monitoring
        performance_monitor = LunaPerformanceMonitor() if LunaPerformanceMonitor else None
        if performance_monitor:
            performance_monitor.start_monitoring("Luna_MediaPipe_Detailer")

        if ENGINE_INSTANCE is None:
            if Mediapipe_Engine is None:
                raise ImportError("Mediapipe_Engine is not available. Please ensure mediapipe_engine.py is properly installed.")
            ENGINE_INSTANCE = Mediapipe_Engine()

        # 1. Generate Mask
        np_image = (image[0].numpy() * 255).astype(np.uint8)
        options = {'model_type': model_type, 'sort_by': sort_by, 'max_objects': max_objects, 'confidence': confidence, 'mask_padding': mask_padding, 'mask_blur': mask_blur}
        final_mask_np = ENGINE_INSTANCE.process_and_create_mask(np_image, options)
        mask_tensor = torch.from_numpy(final_mask_np.astype(np.float32) / 255.0).unsqueeze(0)

        # 2. Encode Prompts with pooled_output for FLUX compatibility
        def encode_text(text):
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            return [[cond, {"pooled_output": pooled}]]

        positive_cond = encode_text(detail_positive_prompt)
        negative_cond = encode_text(detail_negative_prompt)

        # 3. Apply FluxGuidance
        if flux_guidance_strength > 0:
            positive_cond[0][1]['guidance_strength'] = flux_guidance_strength

        # 4. Create Masked Latent and Final Conditionings
        pixels = image
        mask = mask_tensor.unsqueeze(0)

        x = (pixels.shape[2] // 8) * 8
        y = (pixels.shape[1] // 8) * 8
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

        orig_pixels = pixels
        pixels = orig_pixels.clone()
        if pixels.shape[2] != x or pixels.shape[1] != y:
            x_offset = (pixels.shape[2] % 8) // 2
            y_offset = (pixels.shape[1] % 8) // 2
            pixels = pixels[:, y_offset:y + y_offset, x_offset:x + x_offset, :]
            mask = mask[:, :, y_offset:y + y_offset, x_offset:x + x_offset]

        m = (1.0 - mask.round()).squeeze(1)
        for i in range(3):
            pixels[:, :, :, i] -= 0.5
            pixels[:, :, :, i] *= m
            pixels[:, :, :, i] += 0.5

        concat_latent = vae.encode(pixels)
        orig_latent = vae.encode(orig_pixels)

        # Correctly handle the noise_mask toggle
        out_latent = {"samples": orig_latent}
        if noise_mask:
            out_latent["noise_mask"] = mask

        out_pos = node_helpers.conditioning_set_values(positive_cond, {"concat_latent_image": concat_latent, "concat_mask": mask})
        out_neg = node_helpers.conditioning_set_values(negative_cond, {"concat_latent_image": concat_latent, "concat_mask": mask})

        # 5. Create Preview
        inverted_mask_preview = 1.0 - mask_tensor.unsqueeze(-1)
        masked_preview = image * inverted_mask_preview

        # Collect performance stats
        performance_stats = {}
        if performance_monitor:
            performance_stats = performance_monitor.stop_monitoring(
                model_type=model_type,
                sort_by=sort_by,
                max_objects=max_objects,
                confidence=confidence,
                mask_padding=mask_padding,
                mask_blur=mask_blur,
                noise_mask=noise_mask,
                flux_guidance_strength=flux_guidance_strength,
                image_shape=str(image.shape),
                mask_shape=str(mask_tensor.shape)
            )

        return (out_pos, out_neg, out_latent, denoise, mask_tensor, image, clip, vae, masked_preview, performance_stats)
    
NODE_CLASS_MAPPINGS = {
    "Luna_MediaPipe_Detailer": Luna_MediaPipe_Detailer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Luna_MediaPipe_Detailer": "Luna MediaPipe Detailer"
}