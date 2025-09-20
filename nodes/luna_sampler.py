import os
import torch
import comfy.sd
from comfy.cli_args import args
import folder_paths
import nodes
import psutil
import time
from typing import Dict, Any, Tuple, Optional
import numpy as np

# Import Luna validation system
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validation import luna_validator, validate_node_input


class LunaSampler:
    """
    Advanced KSampler with Luna Collection optimizations and monitoring
    """
    CATEGORY = "Luna/Sampling"
    RETURN_TYPES = ("LATENT", "PERFORMANCE_STATS")
    RETURN_NAMES = ("sampled_latent", "performance_stats")
    FUNCTION = "sample"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "luna_pipe": ("LUNA_PIPE", {"tooltip": "Luna pipeline containing model, clip, vae, conditionings, seed, sampler, scheduler"}),
                "latent_image": ("LATENT", {"tooltip": "Input latent image to sample"}),
            },
            "optional": {
                "parameters_pipe": ("PARAMETERS_PIPE", {"tooltip": "Parameters pipeline with steps, cfg, denoise, sampler, scheduler, seed"}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 1000, "tooltip": "Number of sampling steps (override parameters_pipe)"}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Classifier-Free Guidance scale (override parameters_pipe)"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoising strength (override parameters_pipe)"}),
                "enable_adaptive_sampling": ("BOOLEAN", {"default": False, "tooltip": "Enable adaptive sampling based on content analysis"}),
                "adaptive_threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Threshold for adaptive sampling decisions"}),
                "enable_performance_monitoring": ("BOOLEAN", {"default": True, "tooltip": "Enable detailed performance monitoring"}),
                "memory_optimization": ("BOOLEAN", {"default": True, "tooltip": "Enable VRAM optimization techniques"}),
                "batch_optimization": ("BOOLEAN", {"default": False, "tooltip": "Enable batch processing optimizations"}),
            }
        }

    def __init__(self):
        self.performance_history = []
        self.vram_monitor = VRAMMonitor()

    @validate_node_input('steps', min_value=1, max_value=1000)
    @validate_node_input('cfg', min_value=0.0, max_value=100.0)
    @validate_node_input('denoise', min_value=0.0, max_value=1.0)
    @validate_node_input('adaptive_threshold', min_value=0.0, max_value=1.0)
    def sample(self, luna_pipe, latent_image, parameters_pipe=None, steps=None, cfg=None, denoise=None,
               enable_adaptive_sampling=False, adaptive_threshold=0.8,
               enable_performance_monitoring=True, memory_optimization=True,
               batch_optimization=False):

        # Unpack luna_pipe
        model, clip, vae, positive, negative, seed, sampler_name, scheduler_name = luna_pipe

        # Use parameters_pipe if provided, otherwise use individual parameters
        if parameters_pipe is not None:
            param_steps, param_cfg, param_denoise, param_sampler, param_scheduler, param_seed = parameters_pipe
            # Override with provided parameters if specified
            steps = steps if steps is not None else param_steps
            cfg = cfg if cfg is not None else param_cfg
            denoise = denoise if denoise is not None else param_denoise
            sampler_name = param_sampler
            scheduler_name = param_scheduler
            seed = param_seed
        else:
            # Use defaults if no parameters_pipe and no overrides
            steps = steps or 25
            cfg = cfg or 7.0
            denoise = denoise or 1.0

        start_time = time.time()
        initial_vram = self.vram_monitor.get_vram_usage() if enable_performance_monitoring else 0

        try:
            # Adaptive sampling logic
            if enable_adaptive_sampling:
                steps, cfg = self._adaptive_sampling_logic(
                    latent_image, positive, negative, steps, cfg, adaptive_threshold
                )

            # Memory optimization
            if memory_optimization:
                self._optimize_memory_usage(model, latent_image)

            # Batch optimization for multiple latents
            if batch_optimization and isinstance(latent_image, list):
                result = self._batch_sample(model, seed, steps, cfg, sampler_name,
                                          scheduler_name, positive, negative, latent_image, denoise)
            else:
                # Standard sampling using ComfyUI's common_ksampler
                result = self._standard_sample(model, seed, steps, cfg, sampler_name,
                                             scheduler_name, positive, negative, latent_image, denoise)

            # Performance monitoring
            if enable_performance_monitoring:
                performance_stats = self._collect_performance_stats(
                    start_time, initial_vram, result, steps, cfg, sampler_name, scheduler_name
                )
                self.performance_history.append(performance_stats)
            else:
                performance_stats = {}

            return (result, performance_stats)

        except Exception as e:
            error_stats = {
                "error": str(e),
                "sampling_time": time.time() - start_time,
                "sampler": sampler_name,
                "scheduler": scheduler_name,
                "steps": steps,
                "cfg": cfg
            }
            return (latent_image, error_stats)  # Return original latent on error

    def _adaptive_sampling_logic(self, latent_image, positive, negative, steps, cfg, threshold):
        """Analyze conditioning and latent to adapt sampling parameters"""
        # Analyze conditioning strength
        pos_strength = self._analyze_conditioning_strength(positive)
        neg_strength = self._analyze_conditioning_strength(negative)

        # Adjust steps based on conditioning complexity
        conditioning_ratio = pos_strength / max(neg_strength, 0.1)
        if conditioning_ratio > threshold:
            adaptive_steps = min(int(steps * 1.2), 100)  # Increase steps for complex prompts
        elif conditioning_ratio < (1 - threshold):
            adaptive_steps = max(int(steps * 0.8), 10)   # Decrease steps for simple prompts
        else:
            adaptive_steps = steps

        # Adjust CFG based on latent noise level
        latent_noise = self._estimate_latent_noise(latent_image)
        if latent_noise > 0.7:
            adaptive_cfg = min(cfg * 1.1, 20.0)  # Higher CFG for noisy latents
        elif latent_noise < 0.3:
            adaptive_cfg = max(cfg * 0.9, 3.0)   # Lower CFG for clean latents
        else:
            adaptive_cfg = cfg

        return adaptive_steps, adaptive_cfg

    def _analyze_conditioning_strength(self, conditioning):
        """Analyze the strength/complexity of conditioning"""
        if not conditioning or len(conditioning) == 0:
            return 0.0

        # Simple heuristic: analyze tensor magnitude and variance
        cond_tensor = conditioning[0][0] if isinstance(conditioning, list) else conditioning[0]
        magnitude = torch.norm(cond_tensor).item()
        variance = torch.var(cond_tensor).item()

        # Normalize and combine metrics
        strength = min(magnitude / 100.0, 1.0) * (1 + variance / 10.0)
        return min(strength, 1.0)

    def _estimate_latent_noise(self, latent_image):
        """Estimate noise level in latent image"""
        latent_tensor = latent_image["samples"]
        noise_level = torch.std(latent_tensor).item()
        # Normalize to 0-1 range (rough estimation)
        return min(noise_level / 5.0, 1.0)

    def _optimize_memory_usage(self, model, latent_image):
        """Apply memory optimization techniques"""
        # Force garbage collection
        import gc
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()

    def _batch_sample(self, model, seed, steps, cfg, sampler_name, scheduler_name,
                     positive, negative, latent_images, denoise):
        """Optimized batch sampling for multiple latents"""
        results = []

        for i, latent_image in enumerate(latent_images):
            # Use different seed for each batch item
            batch_seed = seed + i

            result = self._standard_sample(model, batch_seed, steps, cfg, sampler_name,
                                         scheduler_name, positive, negative, latent_image, denoise)
            results.append(result)

        # Combine results
        if len(results) == 1:
            return results[0]
        else:
            # Stack latents for batch output
            stacked_samples = torch.stack([r["samples"] for r in results])
            return {"samples": stacked_samples}

    def _standard_sample(self, model, seed, steps, cfg, sampler_name, scheduler_name,
                        positive, negative, latent_image, denoise):
        """Standard sampling using ComfyUI's common_ksampler"""
        # Use ComfyUI's common_ksampler function
        return nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler_name,
                                   positive, negative, latent_image, denoise=denoise)

    def _collect_performance_stats(self, start_time, initial_vram, result, steps, cfg, sampler_name, scheduler_name):
        """Collect comprehensive performance statistics"""
        end_time = time.time()
        final_vram = self.vram_monitor.get_vram_usage()

        stats = {
            "sampling_time": end_time - start_time,
            "vram_usage_mb": final_vram,
            "vram_delta_mb": final_vram - initial_vram,
            "sampler": sampler_name,
            "scheduler": scheduler_name,
            "steps": steps,
            "cfg": cfg,
            "latent_shape": str(result["samples"].shape) if "samples" in result else "unknown",
            "timestamp": end_time,
            "system_memory_percent": psutil.virtual_memory().percent,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
        }

        # Add GPU info if available
        if torch.cuda.is_available():
            stats["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            stats["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024

        return stats


class VRAMMonitor:
    """Monitor VRAM usage across different GPU setups"""

    def get_vram_usage(self):
        """Get current VRAM usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0

    def get_vram_total(self):
        """Get total VRAM in MB"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        return 0


NODE_CLASS_MAPPINGS = {
    "LunaSampler": LunaSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaSampler": "Luna Sampler (Optimized)",
}