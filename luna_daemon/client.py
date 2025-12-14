"""
Luna Daemon Client Library
Used by ComfyUI nodes to communicate with the shared VAE/CLIP daemon.

The daemon loads models on-demand from the first workflow request,
then shares them across all ComfyUI instances.

Component-based architecture:
- CLIP components (clip_l, clip_g, t5xxl) can be shared across model families
- VAE components are family-specific (sdxl_vae, flux_vae, etc.)

v1.3 Features:
- Split daemon support (separate CLIP and VAE daemons)
- CUDA IPC for same-GPU zero-copy tensor transfer
- Length-prefix protocol for efficient serialization
"""

import socket
import pickle
import struct
import torch
import os
from typing import Tuple, Optional, Any, List, Dict
from .config import DAEMON_HOST, DAEMON_PORT, DAEMON_VAE_PORT, CLIENT_TIMEOUT, ENABLE_CUDA_IPC


class DaemonConnectionError(Exception):
    """Raised when daemon is not available"""
    pass


class ModelMismatchError(Exception):
    """Raised when workflow tries to use a different model than what's loaded in daemon"""
    pass


def get_local_gpu_id() -> Optional[int]:
    """Get the GPU ID that the current process is using, if any."""
    if not torch.cuda.is_available():
        return None
    try:
        # Get the current CUDA device
        return torch.cuda.current_device()
    except:
        return None


class DaemonClient:
    """Client for communicating with Luna VAE/CLIP Daemon"""
    
    def __init__(self, host: str = DAEMON_HOST, port: int = DAEMON_PORT):
        self.host = host
        self.port = port
        self.timeout = CLIENT_TIMEOUT
        
        # IPC state (negotiated per-connection for VAE)
        self._ipc_enabled = False
        self._daemon_gpu_id: Optional[int] = None
        self._local_gpu_id: Optional[int] = None
        
        # Automatically attempt IPC negotiation if enabled
        # This enables zero-copy CUDA shared memory for same-GPU VAE operations
        if ENABLE_CUDA_IPC:
            try:
                self.negotiate_ipc()
            except:
                pass  # Daemon might not be running yet, will work lazily
    
    def _send_request(self, request: dict) -> Any:
        """Send request to daemon with Length-Prefix Protocol (Optimized)"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))
            
            # Serialize and send with length prefix
            data = pickle.dumps(request)
            sock.sendall(struct.pack('>I', len(data)) + data)
            
            # Receive response header (4 bytes = uint32 length)
            header = b""
            while len(header) < 4:
                chunk = sock.recv(4 - len(header))
                if not chunk:
                    raise DaemonConnectionError("Connection closed while reading header")
                header += chunk
            
            response_len = struct.unpack('>I', header)[0]
            
            # Receive exact response payload (no slow accumulator)
            chunks = []
            bytes_recd = 0
            while bytes_recd < response_len:
                chunk_size = min(response_len - bytes_recd, 1048576)  # 1MB chunks
                chunk = sock.recv(chunk_size)
                if not chunk:
                    raise DaemonConnectionError("Connection closed while reading response")
                chunks.append(chunk)
                bytes_recd += len(chunk)
            
            response_data = b"".join(chunks)
            sock.close()
            
            result = pickle.loads(response_data)
            
            # Check for error response
            if isinstance(result, dict) and "error" in result:
                # Check if it's a model mismatch error
                if result.get("type") == "model_mismatch":
                    raise ModelMismatchError(result["error"])
                raise DaemonConnectionError(f"Daemon error: {result['error']}")
            
            return result
            
        except ConnectionRefusedError:
            raise DaemonConnectionError(
                "Luna VAE/CLIP Daemon is not running!\n"
                "Start it with: python -m custom_nodes.ComfyUI-Luna-Collection.luna_daemon.server"
            )
        except socket.timeout:
            raise DaemonConnectionError(
                f"Daemon request timed out after {self.timeout}s"
            )
        except Exception as e:
            raise DaemonConnectionError(f"Daemon communication error: {e}")
    
    def is_running(self) -> bool:
        """Check if daemon is available (uses short timeout for health check)"""
        # Retry up to 3 times with small delays to handle startup race conditions
        import time
        for attempt in range(3):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)  # Short timeout for health check, not 120 seconds
                sock.connect((self.host, self.port))
                
                # Send health check request
                request = {"cmd": "health"}
                data = pickle.dumps(request)
                sock.sendall(struct.pack('>I', len(data)) + data)
                
                # Receive response header (4 bytes = uint32 length)
                header = sock.recv(4)
                if len(header) < 4:
                    sock.close()
                    if attempt < 2:
                        time.sleep(0.1)
                    continue
                
                response_len = struct.unpack('>I', header)[0]
                response_data = sock.recv(response_len)
                sock.close()
                
                result = pickle.loads(response_data)
                is_ok = result.get("status") == "ok"
                # Only log on first successful connection after being down, not every check
                # This prevents spam from the 1-second panel refresh
                return is_ok
            except Exception as e:
                if attempt < 2:
                    time.sleep(0.1)  # Small delay before retry
                elif attempt == 2:
                    # Only log failures on final attempt
                    pass  # Silently fail - caller can handle it
        
        return False
    
    def get_info(self) -> dict:
        """Get daemon info (device, VRAM usage, loaded models/components)"""
        return self._send_request({"cmd": "info"})
    
    def negotiate_ipc(self) -> bool:
        """
        Negotiate CUDA IPC mode with daemon.
        
        If both client and daemon are on the same GPU, enables zero-copy
        tensor transfer using CUDA shared memory.
        
        Returns:
            True if IPC mode was enabled, False otherwise
        """
        if not ENABLE_CUDA_IPC:
            return False
        
        self._local_gpu_id = get_local_gpu_id()
        if self._local_gpu_id is None:
            return False
        
        try:
            result = self._send_request({
                "cmd": "negotiate_ipc",
                "client_gpu_id": self._local_gpu_id
            })
            
            if result.get("ipc_enabled"):
                self._daemon_gpu_id = result.get("daemon_gpu_id")
                self._ipc_enabled = True
                return True
        except:
            pass
        
        return False
    
    @property
    def ipc_enabled(self) -> bool:
        """Check if IPC mode is active."""
        return self._ipc_enabled
    
    def unload_models(self) -> dict:
        """Unload all models from daemon to allow loading different ones"""
        return self._send_request({"cmd": "unload"})
    
    def shutdown(self) -> dict:
        """Shutdown the daemon server"""
        return self._send_request({"cmd": "shutdown"})
    
    def register_checkpoint(self, instance_id: str, name: str, path: str, 
                           size_mb: float, device: str, dtype: str) -> dict:
        """
        Register a checkpoint/UNet model with the daemon for tracking.
        
        The daemon doesn't load the model, just tracks it for monitoring purposes.
        
        Args:
            instance_id: Unique identifier for this ComfyUI instance (e.g., "comfyui:8188")
            name: Model name
            path: Path to the checkpoint file
            size_mb: Size in MB
            device: Device where model is loaded (e.g., "cuda:0")
            dtype: Data type (e.g., "fp16", "fp8_e4m3fn")
        
        Returns:
            Dict with registration status
        """
        return self._send_request({
            "cmd": "register_checkpoint",
            "instance_id": instance_id,
            "name": name,
            "path": path,
            "size_mb": size_mb,
            "device": device,
            "dtype": dtype
        })
    
    def unregister_checkpoint(self, instance_id: str) -> dict:
        """Unregister a checkpoint from tracking"""
        return self._send_request({
            "cmd": "unregister_checkpoint",
            "instance_id": instance_id
        })
    
    # =========================================================================
    # Model Registration (new component-based API)
    # =========================================================================
    
    def register_vae(self, vae: Any, vae_type: str) -> dict:
        """
        Register a VAE with the daemon.
        
        The daemon will extract and store the VAE state dict for future use.
        If a VAE of this type is already loaded, validates it matches.
        
        Args:
            vae: The VAE object from checkpoint loader
            vae_type: Type string ('sdxl', 'flux', 'sd3', 'sd15')
        
        Returns:
            Dict with registration status
        """
        # Extract state dict from VAE
        try:
            if hasattr(vae, 'first_stage_model'):
                state_dict = vae.first_stage_model.state_dict()
            else:
                state_dict = vae.state_dict()
        except Exception as e:
            raise DaemonConnectionError(f"Failed to extract VAE state dict: {e}")
        
        return self._send_request({
            "cmd": "register_vae",
            "vae_type": vae_type,
            "state_dict": state_dict
        })
    
    def register_clip(self, clip: Any, clip_type: str) -> dict:
        """
        Register CLIP with the daemon.
        
        The daemon will extract and store the CLIP components for future use.
        Components that are already loaded will be reused (shared).
        
        Args:
            clip: The CLIP object from checkpoint loader
            clip_type: Type string ('sdxl', 'flux', 'sd3', 'sd15')
        
        Returns:
            Dict with registration status and which components were loaded/shared
        """
        # Extract CLIP components based on type
        components = {}
        
        try:
            cond_model = getattr(clip, 'cond_stage_model', None)
            if cond_model is None and hasattr(clip, 'patcher'):
                cond_model = getattr(clip.patcher, 'model', None)
            
            if cond_model is not None:
                # Extract individual components
                if hasattr(cond_model, 'clip_l'):
                    components['clip_l'] = cond_model.clip_l.state_dict()
                if hasattr(cond_model, 'clip_g'):
                    components['clip_g'] = cond_model.clip_g.state_dict()
                if hasattr(cond_model, 't5xxl'):
                    components['t5xxl'] = cond_model.t5xxl.state_dict()
                
                # For SD1.5 which just has a single CLIP
                if not components and hasattr(cond_model, 'state_dict'):
                    components['clip_l'] = cond_model.state_dict()
        except Exception as e:
            raise DaemonConnectionError(f"Failed to extract CLIP components: {e}")
        
        return self._send_request({
            "cmd": "register_clip",
            "clip_type": clip_type,
            "components": components
        })
    
    def register_vae_by_path(self, vae_path: str, vae_type: str) -> dict:
        """
        Register a VAE by path for daemon disk loading.
        
        The daemon loads the VAE from disk directly, avoiding socket 
        serialization of the full state dict.
        
        Args:
            vae_path: Full path to VAE file (.safetensors)
            vae_type: Type string ('sdxl', 'flux', 'sd3', 'sd15')
        
        Returns:
            Dict with registration status
        """
        return self._send_request({
            "cmd": "register_vae_by_path",
            "vae_path": vae_path,
            "vae_type": vae_type
        })
    
    def register_clip_by_path(self, clip_components: dict, model_type: str, clip_type: str) -> dict:
        """
        Register CLIP by paths for daemon disk loading.
        
        The daemon loads the CLIP models from disk directly, avoiding 
        socket serialization of full state dicts.
        
        Args:
            clip_components: Dict of {component_type: path} (e.g. {"clip_l": "path/to/clip_l.safetensors"})
            model_type: Luna model type ("SD1.5", "SDXL", "Flux", etc.)
            clip_type: ComfyUI CLIPType string ("stable_diffusion", "flux", "sd3", etc.)
        
        Returns:
            Dict with registration status
        """
        return self._send_request({
            "cmd": "register_clip_by_path",
            "clip_components": clip_components,
            "model_type": model_type,
            "clip_type": clip_type
        })
    
    # =========================================================================
    # VAE Operations
    # =========================================================================
    
    def vae_encode(self, pixels: torch.Tensor, vae_type: str,
                   tiled: bool = False, tile_size: int = 512, 
                   overlap: int = 64) -> torch.Tensor:
        """
        Encode image pixels to latent space via daemon.
        
        Uses CUDA IPC if available (same GPU), otherwise falls back to pickle.
        Supports tiled encoding for large images with automatic OOM fallback.
        
        Args:
            pixels: Image tensor in ComfyUI format (B, H, W, C), float32, 0-1 range
            vae_type: VAE type string ('sdxl', 'flux', etc.)
            tiled: If True, use tiled encoding for large images
            tile_size: Size of tiles for tiled encoding (pixels)
            overlap: Overlap between tiles (pixels)
        
        Returns:
            Latent tensor
        """
        if self._ipc_enabled and pixels.is_cuda and not tiled:
            # IPC mode doesn't support tiling yet - fallback to regular
            return self._vae_encode_ipc(pixels, vae_type)
        
        # CRITICAL: Detach and move to CPU, then delete GPU version to prevent memory leak
        pixels_cpu = pixels.detach().cpu()
        
        result = self._send_request({
            "cmd": "vae_encode",
            "pixels": pixels_cpu,
            "vae_type": vae_type,
            "tiled": tiled,
            "tile_size": tile_size,
            "overlap": overlap
        })
        
        # Explicitly delete CPU copy after sending
        del pixels_cpu
        return result
    
    def _vae_encode_ipc(self, pixels: torch.Tensor, vae_type: str) -> torch.Tensor:
        """VAE encode using CUDA IPC (zero-copy for same-GPU)."""
        # Move to shared memory
        pixels_shared = pixels.share_memory_()
        
        # Send just the metadata + storage handle
        result = self._send_request({
            "cmd": "vae_encode_ipc",
            "pixels_storage": pixels_shared.storage(),
            "pixels_shape": list(pixels.shape),
            "pixels_dtype": str(pixels.dtype),
            "vae_type": vae_type
        })
        
        # Result is already a CUDA tensor from shared memory
        return result
    
    def vae_decode(self, latents: torch.Tensor, vae_type: str,
                   tiled: bool = False, tile_size: int = 64,
                   overlap: int = 16) -> torch.Tensor:
        """
        Decode latents to image pixels via daemon.
        
        Uses CUDA IPC if available (same GPU), otherwise falls back to pickle.
        Supports tiled decoding for large latents with automatic OOM fallback.
        
        Args:
            latents: Latent tensor
            vae_type: VAE type string ('sdxl', 'flux', etc.)
            tiled: If True, use tiled decoding for large latents
            tile_size: Size of tiles for tiled decoding (latent space)
            overlap: Overlap between tiles (latent space)
        
        Returns:
            Image tensor in ComfyUI format (B, H, W, C), float32, 0-1 range
        """
        if self._ipc_enabled and latents.is_cuda and not tiled:
            # IPC mode doesn't support tiling yet - fallback to regular
            return self._vae_decode_ipc(latents, vae_type)
        
        # CRITICAL: Detach and move to CPU, then delete GPU version to prevent memory leak
        # The daemon will create its own GPU copy, so keeping the original serves no purpose
        latents_cpu = latents.detach().cpu()
        
        result = self._send_request({
            "cmd": "vae_decode",
            "latents": latents_cpu,
            "vae_type": vae_type,
            "tiled": tiled,
            "tile_size": tile_size,
            "overlap": overlap
        })
        
        # Explicitly delete CPU copy after sending to free memory
        del latents_cpu
        return result
    
    def _vae_decode_ipc(self, latents: torch.Tensor, vae_type: str) -> torch.Tensor:
        """VAE decode using CUDA IPC (zero-copy for same-GPU)."""
        # Move to shared memory
        latents_shared = latents.share_memory_()
        
        # Send just the metadata + storage handle
        result = self._send_request({
            "cmd": "vae_decode_ipc",
            "latents_storage": latents_shared.storage(),
            "latents_shape": list(latents.shape),
            "latents_dtype": str(latents.dtype),
            "vae_type": vae_type
        })
        
        # Result is already a CUDA tensor from shared memory
        return result
    
    # =========================================================================
    # CLIP Operations
    # =========================================================================
    
    def clip_encode(
        self, 
        positive: str, 
        negative: str,
        clip_type: str,
        lora_stack: Optional[List[Dict]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode text prompts to CLIP conditioning via daemon.
        
        Args:
            positive: Positive prompt text
            negative: Negative prompt text
            clip_type: CLIP type string ('sdxl', 'flux', 'sd3', 'sd15')
            lora_stack: Optional list of {"hash": str, "strength": float} for LoRA application
        
        Returns:
            Tuple of (cond, pooled, uncond, pooled_neg)
        """
        request: Dict[str, Any] = {
            "cmd": "clip_encode",
            "positive": positive,
            "negative": negative,
            "clip_type": clip_type
        }
        if lora_stack:
            request["lora_stack"] = lora_stack
        
        return self._send_request(request)
    
    def clip_encode_sdxl(
        self,
        positive: str,
        negative: str,
        clip_type: str = "sdxl",
        width: int = 1024,
        height: int = 1024,
        crop_w: int = 0,
        crop_h: int = 0,
        target_width: int = 1024,
        target_height: int = 1024,
        lora_stack: Optional[List[Dict]] = None
    ) -> Tuple[list, list]:
        """
        Encode text prompts with SDXL-specific conditioning (includes size embeddings).
        
        Args:
            positive: Positive prompt text
            negative: Negative prompt text
            clip_type: CLIP type string
            width, height: Original image dimensions
            crop_w, crop_h: Crop coordinates
            target_width, target_height: Target generation size
            lora_stack: Optional list of {"hash": str, "strength": float} for LoRA application
        
        Returns:
            Tuple of (positive_conditioning, negative_conditioning) ready for KSampler
        """
        request: Dict[str, Any] = {
            "cmd": "clip_encode_sdxl",
            "positive": positive,
            "negative": negative,
            "clip_type": clip_type,
            "width": width,
            "height": height,
            "crop_w": crop_w,
            "crop_h": crop_h,
            "target_width": target_width,
            "target_height": target_height
        }
        if lora_stack:
            request["lora_stack"] = lora_stack
        
        return self._send_request(request)
    
    # =========================================================================
    # LoRA Operations (F-150 Architecture)
    # =========================================================================
    
    def has_lora(self, lora_hash: str) -> bool:
        """Check if a LoRA is cached in the daemon's registry."""
        result = self._send_request({
            "cmd": "has_lora",
            "lora_hash": lora_hash
        })
        return result.get("exists", False)
    
    def upload_lora(self, lora_hash: str, weights: Dict[str, torch.Tensor]) -> dict:
        """
        Upload LoRA weights to daemon's registry.
        
        Args:
            lora_hash: Unique hash identifying this LoRA
            weights: Dict of layer_key -> tensor
        
        Returns:
            Dict with upload status
        """
        # Convert tensors to CPU for transport
        cpu_weights = {k: v.cpu() for k, v in weights.items()}
        
        return self._send_request({
            "cmd": "upload_lora",
            "lora_hash": lora_hash,
            "weights": cpu_weights
        })
    
    def register_lora(self, lora_name: str, clip_strength: float = 1.0) -> dict:
        """
        Register a LoRA by name for daemon disk loading.
        
        The daemon loads the LoRA from disk using folder_paths, extracts
        CLIP weights, and caches them. This avoids socket serialization.
        
        Args:
            lora_name: LoRA filename (e.g., "my_lora.safetensors")
            clip_strength: CLIP weight strength
        
        Returns:
            Dict with: hash (content hash for lora_stack), success, size_mb
        """
        return self._send_request({
            "cmd": "register_lora",
            "lora_name": lora_name,
            "clip_strength": clip_strength
        })
    
    def get_lora_stats(self) -> dict:
        """Get statistics about cached LoRAs in daemon."""
        return self._send_request({"cmd": "lora_stats"})
    
    def clear_lora_cache(self) -> dict:
        """Clear all cached LoRAs from daemon."""
        return self._send_request({"cmd": "clear_loras"})
    
    # =========================================================================
    # Z-IMAGE / Qwen3-VL Operations
    # =========================================================================
    
    def zimage_encode(self, text: str) -> torch.Tensor:
        """
        Encode text using Z-IMAGE's Qwen3 encoder.
        
        Routes to daemon's Qwen3-VL service which provides embeddings
        compatible with Z-IMAGE (vocab_size=151936, hidden_size=2560).
        
        Args:
            text: Input prompt text
            
        Returns:
            Conditioning tensor [B, seq_len, 2560]
        """
        return self._send_request({
            "cmd": "zimage_encode",
            "text": text
        })
    
    def zimage_encode_batch(self, texts: list) -> torch.Tensor:
        """
        Batch encode multiple texts for Z-IMAGE.
        
        Args:
            texts: List of prompt strings
            
        Returns:
            Stacked conditioning tensors [B, seq_len, 2560]
        """
        return self._send_request({
            "cmd": "zimage_encode_batch",
            "texts": texts
        })
    
    def describe_image(self, image: torch.Tensor, prompt: str = "Describe this image.") -> str:
        """
        Generate an image description using Qwen3-VL.
        
        Args:
            image: Image tensor (B, H, W, C) or (H, W, C)
            prompt: Instruction for the VLM
            
        Returns:
            Generated description text
        """
        return self._send_request({
            "cmd": "describe_image",
            "image": image.cpu() if isinstance(image, torch.Tensor) else image,
            "prompt": prompt
        })
    
    def extract_style(self, image: torch.Tensor) -> str:
        """
        Extract style descriptors from an image.
        
        Args:
            image: Image tensor
            
        Returns:
            Style description for prompts
        """
        return self._send_request({
            "cmd": "extract_style",
            "image": image.cpu() if isinstance(image, torch.Tensor) else image
        })
    
    def register_qwen3(self, model_path: str, mmproj_path: Optional[str] = None) -> dict:
        """
        Register a Qwen3-VL model with the daemon.
        
        Loads the model once in the daemon for shared use across all
        Z-IMAGE CLIP encoding and LLM generation operations.
        
        Args:
            model_path: Path to GGUF model file
            mmproj_path: Optional path to mmproj file for vision
            
        Returns:
            dict with 'success' and 'model_path' keys
        """
        request = {
            "cmd": "register_qwen3",
            "model_path": model_path
        }
        if mmproj_path:
            request["mmproj_path"] = mmproj_path
        return self._send_request(request)
    
    def llm_generate(self, prompt: str, max_tokens: int = 256, 
                     temperature: float = 0.7, system_prompt: Optional[str] = None) -> str:
        """
        Generate text using the daemon's Qwen3-VL model.
        
        Uses the same model instance as CLIP encoding for efficiency.
        
        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            
        Returns:
            Generated text string
        """
        request = {
            "cmd": "llm_generate",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        if system_prompt:
            request["system_prompt"] = system_prompt
        result = self._send_request(request)
        if isinstance(result, dict) and "text" in result:
            return result["text"]
        if isinstance(result, str):
            return result
        return str(result)
    
    def submit_async(self, cmd: str, data: dict) -> dict:
        """
        Submit async job to daemon and return immediately.
        
        Useful for long-running operations (image saving, etc.) that don't block workflow.
        
        Args:
            cmd: Command name (e.g., "save_images_async")
            data: Request data dictionary
        
        Returns:
            Status dict with job_id for tracking
        """
        request = {"cmd": cmd, **data}
        result = self._send_request(request)
        return result if isinstance(result, dict) else {"error": str(result)}
    
    # =========================================================================
    # MODEL (UNet) Operations
    # =========================================================================
    
    def register_model(self, model: Any, model_type: str) -> dict:
        """
        Register a diffusion model (UNet) with the daemon.
        
        The daemon will extract and freeze the model weights for shared inference.
        Model is set to eval mode with requires_grad=False for VRAM optimization.
        
        Args:
            model: The ModelPatcher object from checkpoint loader
            model_type: Type string ('flux', 'sdxl', 'sd15', etc.)
        
        Returns:
            Dict with registration status
        """
        # Extract model state dict
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
                state_dict = model.model.diffusion_model.state_dict()
            elif hasattr(model, 'state_dict'):
                state_dict = model.state_dict()
            else:
                raise ValueError("Cannot extract model state dict")
        except Exception as e:
            raise DaemonConnectionError(f"Failed to extract model state dict: {e}")
        
        # Move to CPU for transport
        cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
        
        return self._send_request({
            "cmd": "register_model",
            "model_type": model_type,
            "state_dict": cpu_state_dict
        })
    
    def register_model_by_path(self, model_path: str, model_type: str) -> dict:
        """
        Register a model by path for daemon disk loading (preferred method).
        
        The daemon loads the model from disk directly, avoiding socket
        serialization overhead. Model is frozen (eval + requires_grad=False).
        
        Args:
            model_path: Full path to checkpoint/unet file
            model_type: Type string ('flux', 'sdxl', 'sd15', etc.)
        
        Returns:
            Dict with registration status
        """
        return self._send_request({
            "cmd": "register_model_by_path",
            "model_path": model_path,
            "model_type": model_type
        })
    
    def model_forward(self, x: torch.Tensor, timesteps: torch.Tensor,
                     context: Optional[torch.Tensor] = None,
                     model_type: str = 'sdxl',
                     lora_stack: Optional[List[tuple]] = None,
                     **kwargs) -> torch.Tensor:
        """
        Execute UNet forward pass through daemon.
        
        This is the main inference call used by samplers during denoising.
        
        Args:
            x: Noisy latents (B, C, H, W)
            timesteps: Timestep tensor (B,)
            context: Conditioning/context tensor (B, seq_len, dim)
            model_type: Model type string
            lora_stack: Optional list of (lora_name, model_str, clip_str) tuples
            **kwargs: Additional model-specific arguments
        
        Returns:
            Denoised output tensor
        """
        # Move tensors to CPU for transport (daemon will move to its GPU)
        x_cpu = x.detach().cpu()
        timesteps_cpu = timesteps.detach().cpu()
        context_cpu = context.detach().cpu() if context is not None else None
        
        request = {
            "cmd": "model_forward",
            "x": x_cpu,
            "timesteps": timesteps_cpu,
            "context": context_cpu,
            "model_type": model_type
        }
        
        if lora_stack:
            request["lora_stack"] = lora_stack
        
        # Add any extra kwargs (control, y, etc.)
        request.update(kwargs)
        
        result = self._send_request(request)
        
        # Move result back to original device
        if isinstance(result, torch.Tensor):
            return result.to(x.device)
        return result


# =============================================================================
# Singleton client instance
# =============================================================================

_client: Optional[DaemonClient] = None


def get_client() -> DaemonClient:
    """Get or create the singleton client instance"""
    global _client
    if _client is None:
        _client = DaemonClient()
    return _client


# =============================================================================
# Convenience functions
# =============================================================================

def is_daemon_running() -> bool:
    """Check if daemon is available"""
    return get_client().is_running()


def start_daemon() -> bool:
    """Start the Luna Daemon Tray if not running (single-instance enforced)"""
    # Check if daemon is already responding
    if is_daemon_running():
        return True
    
    try:
        import subprocess
        import sys
        from pathlib import Path
        
        # Path to tray app (enforces single instance)
        tray_app = Path(__file__).parent / "tray_app.py"
        
        if not tray_app.exists():
            print(f"[LunaDaemon] Tray app not found at {tray_app}")
            return False
        
        # Start tray app - it will handle single-instance checking
        # and auto-start the daemon server
        if sys.platform == "win32":
            # Windows: Start tray app (visible in system tray)
            subprocess.Popen(
                [sys.executable, str(tray_app)],
                creationflags=0,  # Normal window for tray visibility
                cwd=str(tray_app.parent)
            )
        else:
            # Unix: Start tray app in background
            subprocess.Popen(
                [sys.executable, str(tray_app)],
                start_new_session=True,
                cwd=str(tray_app.parent)
            )
        
        print(f"[LunaDaemon] Tray app started, check system tray")
        return True
    except Exception as e:
        print(f"[LunaDaemon] Error starting daemon tray: {e}")
        return False


def get_daemon_info() -> dict:
    """Get daemon info including loaded models"""
    return get_client().get_info()


def register_checkpoint(instance_id: str, name: str, path: str,
                       size_mb: float, device: str, dtype: str) -> dict:
    """Register a checkpoint/UNet model with daemon for tracking"""
    return get_client().register_checkpoint(instance_id, name, path, size_mb, device, dtype)


def unregister_checkpoint(instance_id: str) -> dict:
    """Unregister a checkpoint from daemon tracking"""
    return get_client().unregister_checkpoint(instance_id)


def unload_daemon_models() -> dict:
    """Unload all models from daemon"""
    return get_client().unload_models()


def register_vae(vae: Any, vae_type: str) -> dict:
    """Register VAE with daemon"""
    return get_client().register_vae(vae, vae_type)


def register_clip(clip: Any, clip_type: str) -> dict:
    """Register CLIP with daemon"""
    return get_client().register_clip(clip, clip_type)


def register_vae_by_path(vae_path: str, vae_type: str) -> dict:
    """Register VAE by path with daemon (loads from disk)"""
    return get_client().register_vae_by_path(vae_path, vae_type)


def register_clip_by_path(clip_components: dict, model_type: str, clip_type: str) -> dict:
    """Register CLIP by paths with daemon (loads from disk)

    clip_components: Dict of {component_type: path} (e.g. {"clip_l": "path/to/clip_l.safetensors"})
    """
    return get_client().register_clip_by_path(clip_components, model_type, clip_type)


def vae_encode(pixels: torch.Tensor, vae_type: str,
               tiled: bool = False, tile_size: int = 512,
               overlap: int = 64) -> torch.Tensor:
    """
    Encode pixels via daemon.
    
    Args:
        pixels: Image tensor (B, H, W, C)
        vae_type: VAE type string
        tiled: If True, use tiled encoding for large images
        tile_size: Tile size in pixels
        overlap: Overlap between tiles
        
    Returns:
        Latent tensor
    """
    return get_client().vae_encode(pixels, vae_type, tiled=tiled, 
                                    tile_size=tile_size, overlap=overlap)


def vae_decode(latents: torch.Tensor, vae_type: str,
               tiled: bool = False, tile_size: int = 64,
               overlap: int = 16) -> torch.Tensor:
    """
    Decode latents via daemon.
    
    Args:
        latents: Latent tensor
        vae_type: VAE type string
        tiled: If True, use tiled decoding for large latents
        tile_size: Tile size in latent space
        overlap: Overlap between tiles
        
    Returns:
        Image tensor (B, H, W, C)
    """
    return get_client().vae_decode(latents, vae_type, tiled=tiled,
                                    tile_size=tile_size, overlap=overlap)


def clip_encode(positive: str, negative: str, clip_type: str, lora_stack: Optional[List[Dict]] = None) -> Tuple:
    """Encode text via daemon"""
    return get_client().clip_encode(positive, negative, clip_type, lora_stack=lora_stack)


def clip_encode_sdxl(positive: str, negative: str, clip_type: str = "sdxl", lora_stack: Optional[List[Dict]] = None, **kwargs) -> Tuple[list, list]:
    """Encode text with SDXL conditioning via daemon"""
    return get_client().clip_encode_sdxl(positive, negative, clip_type, lora_stack=lora_stack, **kwargs)


def has_lora(lora_hash: str) -> bool:
    """Check if LoRA is cached in daemon"""
    return get_client().has_lora(lora_hash)


def upload_lora(lora_hash: str, weights: Dict[str, torch.Tensor]) -> dict:
    """Upload LoRA weights to daemon"""
    return get_client().upload_lora(lora_hash, weights)


def register_lora(lora_name: str, clip_strength: float = 1.0) -> dict:
    """Register LoRA by name for daemon disk loading"""
    return get_client().register_lora(lora_name, clip_strength)


def get_lora_stats() -> dict:
    """Get LoRA cache statistics"""
    return get_client().get_lora_stats()


def clear_lora_cache() -> dict:
    """Clear LoRA cache"""
    return get_client().clear_lora_cache()


# =============================================================================
# MODEL (UNet) Operations
# =============================================================================

def register_model(model: Any, model_type: str) -> dict:
    """Register a diffusion model with the daemon"""
    return get_client().register_model(model, model_type)


def register_model_by_path(model_path: str, model_type: str) -> dict:
    """Register a model by path (preferred - avoids socket serialization)"""
    return get_client().register_model_by_path(model_path, model_type)


def model_forward(x: torch.Tensor, timesteps: torch.Tensor,
                 context: Optional[torch.Tensor] = None,
                 model_type: str = 'sdxl',
                 lora_stack: Optional[List[tuple]] = None,
                 **kwargs) -> torch.Tensor:
    """Execute UNet forward pass through daemon"""
    return get_client().model_forward(x, timesteps, context, model_type, lora_stack, **kwargs)


# =============================================================================
# Z-IMAGE / Qwen3-VL Operations
# =============================================================================

def zimage_encode(text: str) -> torch.Tensor:
    """
    Encode text using Z-IMAGE's Qwen3 encoder (via daemon's Qwen3-VL).
    
    This routes to the daemon's Qwen3-VL encoder service, which provides
    embeddings compatible with Z-IMAGE's Qwen3-4B CLIP.
    
    Args:
        text: Input prompt text
        
    Returns:
        Conditioning tensor [B, seq_len, 2560] compatible with Z-IMAGE
    """
    return get_client().zimage_encode(text)


def zimage_encode_batch(texts: list) -> torch.Tensor:
    """
    Batch encode multiple texts for Z-IMAGE.
    
    Efficient when encoding multiple prompts (e.g., for batch generation).
    
    Args:
        texts: List of prompt strings
        
    Returns:
        Stacked conditioning tensors [B, seq_len, 2560]
    """
    return get_client().zimage_encode_batch(texts)


def describe_image(image: torch.Tensor, prompt: str = "Describe this image in detail.") -> str:
    """
    Generate a description of an image using daemon's VLM.
    
    Uses Qwen3-VL's vision-language capabilities.
    
    Args:
        image: Image tensor (B, H, W, C) or (H, W, C)
        prompt: Instruction for the VLM
        
    Returns:
        Generated description text
    """
    return get_client().describe_image(image, prompt)


def extract_style(image: torch.Tensor) -> str:
    """
    Extract style descriptors from an image.
    
    Returns a prompt-style description of the image's artistic style.
    
    Args:
        image: Image tensor
        
    Returns:
        Style description suitable for prompts
    """
    return get_client().extract_style(image)


def is_qwen3_loaded() -> bool:
    """Check if daemon has Qwen3-VL encoder loaded."""
    try:
        info = get_daemon_info()
        return info.get('qwen3_loaded', False)
    except:
        return False


def register_qwen3(model_path: str, mmproj_path: Optional[str] = None) -> dict:
    """
    Register a Qwen3-VL model with the daemon.
    
    Loads the model in the daemon for shared CLIP encoding and LLM generation.
    
    Args:
        model_path: Path to GGUF model file
        mmproj_path: Optional path to mmproj file for vision
        
    Returns:
        dict with 'success' and 'model_path' keys
    """
    return get_client().register_qwen3(model_path, mmproj_path)


def llm_generate(prompt: str, max_tokens: int = 256, 
                 temperature: float = 0.7, system_prompt: Optional[str] = None) -> str:
    """
    Generate text using the daemon's Qwen3-VL model.
    
    Uses the same model as CLIP encoding - no duplicate loading.
    
    Args:
        prompt: Input prompt for generation
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        system_prompt: Optional system prompt
        
    Returns:
        Generated text string
    """
    return get_client().llm_generate(prompt, max_tokens, temperature, system_prompt)


# =============================================================================
# Split Daemon Clients (v1.3)
# =============================================================================

_vae_client: Optional[DaemonClient] = None
_clip_client: Optional[DaemonClient] = None


def get_vae_client(port: int = DAEMON_VAE_PORT, host: str = DAEMON_HOST) -> DaemonClient:
    """
    Get a client specifically for VAE operations.
    
    In split daemon mode, VAE runs on primary GPU with CUDA IPC support.
    Auto-negotiates IPC on first use.
    """
    global _vae_client
    if _vae_client is None:
        _vae_client = DaemonClient(host=host, port=port)
        # Try to negotiate IPC for same-GPU zero-copy
        if ENABLE_CUDA_IPC:
            try:
                if _vae_client.negotiate_ipc():
                    print(f"[LunaClient] VAE IPC enabled (GPU {_vae_client._daemon_gpu_id})")
            except:
                pass
    return _vae_client


def get_clip_client(port: int = DAEMON_PORT, host: str = DAEMON_HOST) -> DaemonClient:
    """
    Get a client specifically for CLIP operations.
    
    In split daemon mode, CLIP runs on secondary GPU.
    """
    global _clip_client
    if _clip_client is None:
        _clip_client = DaemonClient(host=host, port=port)
    return _clip_client


def reset_clients():
    """Reset all client instances (for testing or reconnection)."""
    global _client, _vae_client, _clip_client
    _client = None
    _vae_client = None
    _clip_client = None
