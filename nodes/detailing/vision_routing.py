"""
Vision Routing - Dynamic CLIP-ViT routing for Luna Detailing

Provides intelligent routing for vision encoding:
- If Luna Daemon is available with vision workers → Route to daemon (GPU1)
- Otherwise → Encode locally on GPU0 (no transfer overhead)

This enables optimal VRAM usage:
- Daemon: Full image sent once, daemon crops on GPU, batch encodes
- Local: Direct encoding, no socket/pickle overhead
"""

import torch
from typing import List, Tuple, Optional, Any

# Try to import daemon client
try:
    from ...luna_daemon.client import DaemonClient, DaemonConnectionError
    HAS_DAEMON = True
except ImportError:
    HAS_DAEMON = False
    DaemonClient = None  # type: ignore
    DaemonConnectionError = Exception  # type: ignore


class VisionRouter:
    """
    Routes CLIP-ViT encoding to daemon or local based on availability.
    
    Usage:
        router = VisionRouter(clip_vision)
        embeddings = router.encode_crops(full_image, crop_coords)
    """
    
    def __init__(self, clip_vision: Optional[Any] = None):
        """
        Initialize vision router.
        
        Args:
            clip_vision: Local CLIP_VISION model (optional, for local fallback)
        """
        self.clip_vision = clip_vision
        self._daemon: Optional[Any] = None
        self._daemon_available: Optional[bool] = None
        self._use_daemon: bool = False
        
        # Check daemon availability on init
        self._check_daemon()
    
    def _check_daemon(self) -> bool:
        """Check if daemon is available with vision workers."""
        if not HAS_DAEMON:
            self._daemon_available = False
            self._use_daemon = False
            return False
        
        try:
            self._daemon = DaemonClient()
            if self._daemon.is_running():
                status = self._daemon.vision_status()
                self._daemon_available = bool(status.get("available", False))
                self._use_daemon = self._daemon_available and bool(status.get("loaded", False))
                return self._use_daemon
        except Exception:
            pass
        
        self._daemon_available = False
        self._use_daemon = False
        return False
    
    @property
    def using_daemon(self) -> bool:
        """Check if currently routing to daemon."""
        return self._use_daemon
    
    @property
    def available(self) -> bool:
        """Check if any vision encoding is available (daemon or local)."""
        return self._use_daemon or self.clip_vision is not None
    
    def encode_crops(
        self,
        full_image: torch.Tensor,
        crop_coords: List[Tuple[int, int, int, int]],
        tile_size: int = 1024
    ) -> List[torch.Tensor]:
        """
        Encode crops with CLIP-ViT, routing to daemon or local.
        
        Args:
            full_image: Full canvas [1, H, W, 3] or [H, W, 3] in BHWC
            crop_coords: List of (x1, y1, x2, y2) crop regions
            tile_size: Target size for encoding (default 1024)
        
        Returns:
            List of vision embeddings, one per crop
        """
        if len(crop_coords) == 0:
            return []
        
        if self._use_daemon:
            return self._encode_via_daemon(full_image, crop_coords, tile_size)
        elif self.clip_vision is not None:
            return self._encode_local(full_image, crop_coords, tile_size)
        else:
            raise RuntimeError("No vision encoder available (daemon not running, no local clip_vision)")
    
    def encode_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Batch encode images with CLIP-ViT.
        
        Args:
            images: Batch of images [N, H, W, 3] in BHWC
        
        Returns:
            Vision embeddings [N, 257, vision_dim]
        """
        if self._use_daemon:
            return self._daemon.vision_encode_batch(images)
        elif self.clip_vision is not None:
            return self._encode_batch_local(images)
        else:
            raise RuntimeError("No vision encoder available")
    
    def _encode_via_daemon(
        self,
        full_image: torch.Tensor,
        crop_coords: List[Tuple[int, int, int, int]],
        tile_size: int
    ) -> List[torch.Tensor]:
        """Route to daemon - sends full image, daemon crops and encodes."""
        try:
            result = self._daemon.vision_encode_crops(
                full_image=full_image,
                crop_coords=crop_coords,
                tile_size=tile_size
            )
            print(f"[VisionRouter] ✓ Daemon encoded {len(result)} crops")
            return result
        except DaemonConnectionError as e:
            print(f"[VisionRouter] ⚠ Daemon connection failed: {e}")
            # Fallback to local if available
            if self.clip_vision is not None:
                print("[VisionRouter] Falling back to local encoding")
                self._use_daemon = False
                return self._encode_local(full_image, crop_coords, tile_size)
            raise
    
    def _encode_local(
        self,
        full_image: torch.Tensor,
        crop_coords: List[Tuple[int, int, int, int]],
        tile_size: int
    ) -> List[torch.Tensor]:
        """Encode locally - crop and batch encode on current device."""
        import torch.nn.functional as F
        
        # Ensure proper shape
        if full_image.dim() == 3:
            full_image = full_image.unsqueeze(0)
        
        device = next(self.clip_vision.model.parameters()).device
        full_image = full_image.to(device)
        
        # Crop on GPU
        crops = []
        for (x1, y1, x2, y2) in crop_coords:
            crop = full_image[:, y1:y2, x1:x2, :]
            
            # Resize to tile_size if needed
            if crop.shape[1] != tile_size or crop.shape[2] != tile_size:
                crop_bchw = crop.permute(0, 3, 1, 2)
                crop_bchw = F.interpolate(
                    crop_bchw, size=(tile_size, tile_size),
                    mode='bicubic', align_corners=False
                )
                crop = crop_bchw.permute(0, 2, 3, 1)
            
            crops.append(crop)
        
        # Stack into batch
        batch = torch.cat(crops, dim=0)
        
        # Batch encode
        with torch.inference_mode():
            output = self.clip_vision.encode_image(batch)
            
            if hasattr(output, 'last_hidden_state'):
                embeddings = output.last_hidden_state
            elif hasattr(output, 'image_embeds'):
                embeddings = output.image_embeds
            else:
                embeddings = output
        
        # Split into list
        result = [embeddings[i:i+1] for i in range(len(crop_coords))]
        
        print(f"[VisionRouter] ✓ Local encoded {len(result)} crops")
        return result
    
    def _encode_batch_local(self, images: torch.Tensor) -> torch.Tensor:
        """Batch encode locally."""
        device = next(self.clip_vision.model.parameters()).device
        images = images.to(device)
        
        with torch.inference_mode():
            output = self.clip_vision.encode_image(images)
            
            if hasattr(output, 'last_hidden_state'):
                return output.last_hidden_state
            elif hasattr(output, 'image_embeds'):
                return output.image_embeds
            return output


def get_vision_router(clip_vision: Optional[Any] = None) -> VisionRouter:
    """Factory function to create a vision router."""
    return VisionRouter(clip_vision)
