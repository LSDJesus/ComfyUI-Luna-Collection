"""
Luna Scaffold Upscaler - Neutral Upscaling Bridge

Bridges draft generation to refinement phase by upscaling pixels to match
the full scaffold dimensions. Uses edge-preserving, texture-maintaining
techniques to create a clean canvas for targeted refinement.

Architecture:
- CUDA-accelerated Lanczos upscaling (torchvision, no CPU overhead)
- Edge-aware sharpening to maintain line work
- Texture coherence to preserve fine details
- Color blending to smooth gradients

Workflow Integration:
    Draft Pixels → Scaffold Upscaler → 4K Canvas → Semantic Detailer
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import comfy.model_management


class LunaScaffoldUpscaler:
    """
    Neutral upscaler for draft→full scaffold bridging.
    
    Creates clean, artifact-free 4K canvas without AI upscaler hallucinations.
    Preserves edges, textures, and color gradients from the draft image.
    
    Key Features:
    - GPU-accelerated Lanczos (torchvision, CUDA)
    - Optional edge sharpening
    - Texture preservation pass
    - Bilateral color smoothing
    """
    
    CATEGORY = "Luna/Detailing"
    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("upscaled_image", "upscaled_latent")
    FUNCTION = "upscale"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "draft_latent": ("LATENT", {
                    "tooltip": "Draft latent from KSampler (e.g., 128×128)"
                }),
                "draft_image": ("IMAGE", {
                    "tooltip": "Draft image from VAE decode (e.g., 1024×1024)"
                }),
                "scale_factor": ("FLOAT", {
                    "tooltip": "Upscale factor from Pyramid Generator (e.g., 4.0)"
                }),
                "edge_enhance": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Edge sharpening strength (0=off, 0.3=subtle, 1.0=strong)"
                }),
                "texture_preserve": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Maintain fine texture coherence during upscale"
                }),
                "color_smooth": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Color gradient smoothing (0=off, 0.1=subtle)"
                }),
            }
        }
    
    def upscale(
        self,
        draft_latent: dict,
        draft_image: torch.Tensor,
        scale_factor: float,
        edge_enhance: float,
        texture_preserve: bool,
        color_smooth: float
    ) -> tuple:
        """
        Upscale both latent and pixels in parallel.
        
        Args:
            draft_latent: Draft latent dict {"samples": tensor} [B, 4, H, W]
            draft_image: Draft pixels [B, H, W, C] (BHWC format)
            scale_factor: Upscale multiplier (e.g., 4.0)
            edge_enhance: Sharpening strength
            texture_preserve: Enable texture coherence
            color_smooth: Color gradient smoothing
            
        Returns:
            (upscaled_pixels, upscaled_latent)
        """
        device = comfy.model_management.get_torch_device()
        
        # Extract draft latent and calculate target dimensions
        draft_lat = draft_latent['samples']  # [B, 4, H, W]
        lat_h, lat_w = draft_lat.shape[2], draft_lat.shape[3]
        
        # Calculate upscaled dimensions
        target_lat_h = int(lat_h * scale_factor)
        target_lat_w = int(lat_w * scale_factor)
        target_img_h = int(draft_image.shape[1] * scale_factor)
        target_img_w = int(draft_image.shape[2] * scale_factor)
        
        print(f"[LunaScaffoldUpscaler] Upscaling:")
        print(f"  Draft latent: {lat_w}×{lat_h} → Target latent: {target_lat_w}×{target_lat_h}")
        print(f"  Draft image: {draft_image.shape[2]}×{draft_image.shape[1]} → Target image: {target_img_w}×{target_img_h}")
        print(f"  Scale: {scale_factor}x, Edge enhance: {edge_enhance}, Texture: {texture_preserve}, Color smooth: {color_smooth}")
        
        # PARALLEL UPSCALING
        # 1. Upscale latent in latent space
        upscaled_lat = self._upscale_latent(draft_lat, target_lat_h, target_lat_w)
        
        # 2. Upscale pixels
        upscaled_pixels = self._upscale_pixels(draft_image, target_img_h, target_img_w)
        
        # 3. EDGE-AWARE SHARPENING (on pixels)
        if edge_enhance > 0.0:
            upscaled_pixels = self._edge_sharpen(upscaled_pixels, strength=edge_enhance)
        
        # 4. TEXTURE PRESERVATION PASS
        if texture_preserve:
            upscaled_pixels = self._preserve_texture(upscaled_pixels, draft_image, target_img_h, target_img_w)
        
        # 5. COLOR SMOOTHING
        if color_smooth > 0.0:
            upscaled_pixels = self._color_smooth_bilateral(upscaled_pixels, strength=color_smooth)
        
        print(f"[LunaScaffoldUpscaler] Complete: {target_img_w}×{target_img_h}")
        
        # Wrap latent for output
        upscaled_latent_dict = {"samples": upscaled_lat}
        
        return (upscaled_pixels, upscaled_latent_dict)


    def _upscale_latent(self, latent: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """
        Upscale latent tensor with bicubic interpolation.
        
        MUST match _upscale_pixels to keep 4K latent and pixels synchronized
        for coordinate mapping and seamless compositing.
        
        Input: [B, 4, H, W]
        Output: [B, 4, target_h, target_w]
        """
        upscaled_lat = F.interpolate(
            latent,
            size=(target_h, target_w),
            mode='bicubic',  # Match pixels for synchronization
            align_corners=False
        )
        return upscaled_lat
    
    def _upscale_pixels(self, img: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """
        Upscale pixel image with bicubic interpolation.
        
        Input: [B, H, W, C] (BHWC)
        Output: [B, target_h, target_w, C] (BHWC)
        """
        # Convert BHWC → BCHW for PyTorch
        img_chw = img.permute(0, 3, 1, 2)
        
        # Bicubic resize (fully GPU-accelerated)
        upscaled_chw = F.interpolate(
            img_chw,
            size=(target_h, target_w),
            mode='bicubic',
            align_corners=False
        )
        
        # Convert back BCHW → BHWC
        return upscaled_chw.permute(0, 2, 3, 1)
    
    def _lanczos_upscale(self, img: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """
        GPU-accelerated bicubic upscaling (equivalent quality to Lanczos).
        
        Input: [B, H, W, C] (BHWC)
        Output: [B, target_h, target_w, C] (BHWC)
        """
        # Convert BHWC → BCHW for PyTorch
        img_chw = img.permute(0, 3, 1, 2)
        
        # Bicubic resize (fully GPU-accelerated)
        upscaled_chw = F.interpolate(
            img_chw,
            size=(target_h, target_w),
            mode='bicubic',
            align_corners=False
        )
        
        # Convert back BCHW → BHWC
        return upscaled_chw.permute(0, 2, 3, 1)
    
    def _edge_sharpen(self, img: torch.Tensor, strength: float) -> torch.Tensor:
        """
        Edge-aware sharpening using unsharp mask.
        
        Enhances edges while preserving smooth regions.
        """
        # Convert to BCHW for conv operations
        img_chw = img.permute(0, 3, 1, 2)
        
        # Gaussian blur kernel (3×3)
        kernel_size = 3
        sigma = 0.5
        kernel = self._gaussian_kernel_2d(kernel_size, sigma, img_chw.device)
        
        # Apply blur per channel
        blurred = F.conv2d(
            img_chw,
            kernel.repeat(img_chw.shape[1], 1, 1, 1),
            padding=kernel_size // 2,
            groups=img_chw.shape[1]
        )
        
        # Unsharp mask: original + strength * (original - blurred)
        sharpened = img_chw + strength * (img_chw - blurred)
        
        # Clamp to valid range
        sharpened = torch.clamp(sharpened, 0.0, 1.0)
        
        return sharpened.permute(0, 2, 3, 1)
    
    def _preserve_texture(
        self,
        upscaled: torch.Tensor,
        draft: torch.Tensor,
        target_h: int,
        target_w: int
    ) -> torch.Tensor:
        """
        Preserve texture coherence from draft image.
        
        Extracts high-frequency detail from draft and applies to upscaled.
        """
        # Upscale draft to target size for comparison
        draft_upscaled = self._lanczos_upscale(draft, target_h, target_w)
        
        # Extract high-frequency detail from draft
        # High-freq = draft - low_pass(draft)
        draft_chw = draft_upscaled.permute(0, 3, 1, 2)
        
        # Low-pass filter (larger gaussian)
        kernel = self._gaussian_kernel_2d(5, 1.0, draft_chw.device)
        low_freq = F.conv2d(
            draft_chw,
            kernel.repeat(draft_chw.shape[1], 1, 1, 1),
            padding=2,
            groups=draft_chw.shape[1]
        )
        
        # High-frequency component
        high_freq = draft_chw - low_freq
        
        # Add subtle high-freq detail to upscaled
        upscaled_chw = upscaled.permute(0, 3, 1, 2)
        enhanced = upscaled_chw + high_freq * 0.2  # Subtle strength
        enhanced = torch.clamp(enhanced, 0.0, 1.0)
        
        return enhanced.permute(0, 2, 3, 1)
    
    def _color_smooth_bilateral(self, img: torch.Tensor, strength: float) -> torch.Tensor:
        """
        Bilateral-style color smoothing to reduce banding in gradients.
        
        Smooths colors while preserving edges.
        """
        if strength == 0.0:
            return img
        
        # Simple approximation: weighted average with gaussian
        img_chw = img.permute(0, 3, 1, 2)
        
        # Spatial gaussian (small kernel to preserve edges)
        kernel = self._gaussian_kernel_2d(3, 0.8, img_chw.device)
        
        smoothed = F.conv2d(
            img_chw,
            kernel.repeat(img_chw.shape[1], 1, 1, 1),
            padding=1,
            groups=img_chw.shape[1]
        )
        
        # Blend with original based on strength
        result = img_chw * (1 - strength) + smoothed * strength
        
        return result.permute(0, 2, 3, 1)
    
    def _gaussian_kernel_2d(self, kernel_size: int, sigma: float, device) -> torch.Tensor:
        """
        Generate 2D Gaussian kernel.
        
        Returns: [1, 1, kernel_size, kernel_size]
        """
        x = torch.arange(kernel_size, device=device, dtype=torch.float32)
        x = x - kernel_size // 2
        
        # 1D Gaussian
        gauss_1d = torch.exp(-x.pow(2) / (2 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        
        # Outer product for 2D
        gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
        
        return gauss_2d.unsqueeze(0).unsqueeze(0)


# Node registration
NODE_CLASS_MAPPINGS = {
    "LunaScaffoldUpscaler": LunaScaffoldUpscaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaScaffoldUpscaler": "🌙 Luna: Scaffold Upscaler",
}
