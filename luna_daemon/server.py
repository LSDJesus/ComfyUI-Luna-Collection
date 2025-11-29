"""
Luna VAE/CLIP Daemon Server
Loads CLIP and VAE models once on a dedicated GPU, serves requests from multiple ComfyUI instances.

Usage:
    python -m custom_nodes.ComfyUI-Luna-Collection.luna_daemon.server
    
Or from ComfyUI root:
    python -m luna_daemon.server
"""

import os
import sys
import socket
import pickle
import threading
import time
import logging
from typing import Any, Dict, Tuple

import torch

# Add ComfyUI to path if needed
comfy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if comfy_path not in sys.path:
    sys.path.insert(0, comfy_path)

from .config import (
    DAEMON_HOST, DAEMON_PORT, SHARED_DEVICE,
    VAE_PATH, CLIP_L_PATH, CLIP_G_PATH, EMBEDDINGS_DIR,
    MAX_WORKERS, LOG_LEVEL
)

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='[%(asctime)s] [Daemon] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class VAECLIPDaemon:
    """
    Daemon server that loads VAE and CLIP models once and serves encode/decode requests.
    """
    
    def __init__(self, device: str = SHARED_DEVICE):
        self.device = device
        self.lock = threading.Lock()
        self.request_count = 0
        self.start_time = None
        
        # Models (loaded on startup)
        self.vae = None
        self.clip = None
        
    def load_models(self):
        """Load VAE and CLIP models to GPU"""
        logger.info(f"Loading models to {self.device}...")
        self.start_time = time.time()
        
        try:
            # Import ComfyUI modules
            import comfy.sd
            import comfy.utils
            import folder_paths
            
            # Load VAE using ComfyUI's loader
            logger.info(f"Loading VAE from: {VAE_PATH}")
            if os.path.exists(VAE_PATH):
                sd = comfy.utils.load_torch_file(VAE_PATH)
                self.vae = comfy.sd.VAE(sd=sd)
                self.current_vae_path = VAE_PATH
                logger.info("VAE loaded successfully")
            else:
                logger.error(f"VAE file not found: {VAE_PATH}")
                raise FileNotFoundError(f"VAE not found: {VAE_PATH}")
            
            # Load CLIP using ComfyUI's loader
            logger.info(f"Loading CLIP from: {CLIP_L_PATH}, {CLIP_G_PATH}")
            clip_paths = []
            if os.path.exists(CLIP_L_PATH):
                clip_paths.append(CLIP_L_PATH)
                self.current_clip_l_path = CLIP_L_PATH
            if os.path.exists(CLIP_G_PATH):
                clip_paths.append(CLIP_G_PATH)
                self.current_clip_g_path = CLIP_G_PATH
            
            if clip_paths:
                self.clip = comfy.sd.load_clip(
                    ckpt_paths=clip_paths,
                    embedding_directory=EMBEDDINGS_DIR if os.path.exists(EMBEDDINGS_DIR) else None
                )
                logger.info("CLIP loaded successfully")
            else:
                logger.error("No CLIP files found!")
                raise FileNotFoundError("CLIP files not found")
            
            # Move models to target device
            # Note: ComfyUI manages device placement internally, but we can hint
            logger.info(f"Models loaded. Moving to {self.device}...")
            
            # Force CUDA cache clear
            torch.cuda.empty_cache()
            
            # Get VRAM usage
            if 'cuda' in self.device:
                device_idx = int(self.device.split(':')[1]) if ':' in self.device else 0
                vram_used = torch.cuda.memory_allocated(device_idx) / 1024**3
                vram_total = torch.cuda.get_device_properties(device_idx).total_memory / 1024**3
                logger.info(f"VRAM usage on {self.device}: {vram_used:.2f} / {vram_total:.2f} GB")
            
            load_time = time.time() - self.start_time
            logger.info(f"All models loaded in {load_time:.1f}s")
            
        except ImportError as e:
            logger.error(f"Failed to import ComfyUI modules: {e}")
            logger.error("Make sure you're running from the ComfyUI directory")
            raise
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def vae_encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode image pixels to latent space"""
        with self.lock:
            try:
                # ComfyUI VAE expects (B, H, W, C) format
                if pixels.dim() == 3:
                    pixels = pixels.unsqueeze(0)
                
                # Encode
                latents = self.vae.encode(pixels)
                
                return latents.cpu()
            except Exception as e:
                logger.error(f"VAE encode error: {e}")
                raise
    
    def vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent space to image pixels"""
        with self.lock:
            try:
                # Decode
                pixels = self.vae.decode(latents)
                
                return pixels.cpu()
            except Exception as e:
                logger.error(f"VAE decode error: {e}")
                raise
    
    def clip_encode(self, positive: str, negative: str = "") -> Tuple:
        """Encode text to CLIP conditioning"""
        with self.lock:
            try:
                # Tokenize and encode positive
                tokens_pos = self.clip.tokenize(positive)
                cond, pooled = self.clip.encode_from_tokens(tokens_pos, return_pooled=True)
                
                # Tokenize and encode negative
                tokens_neg = self.clip.tokenize(negative if negative else "")
                uncond, pooled_neg = self.clip.encode_from_tokens(tokens_neg, return_pooled=True)
                
                return (cond.cpu(), pooled.cpu(), uncond.cpu(), pooled_neg.cpu())
            except Exception as e:
                logger.error(f"CLIP encode error: {e}")
                raise
    
    def clip_encode_sdxl(
        self,
        positive: str,
        negative: str = "",
        width: int = 1024,
        height: int = 1024,
        crop_w: int = 0,
        crop_h: int = 0,
        target_width: int = 1024,
        target_height: int = 1024
    ) -> Tuple[list, list]:
        """Encode text with SDXL-specific size conditioning"""
        with self.lock:
            try:
                # Tokenize
                tokens_pos = self.clip.tokenize(positive)
                tokens_neg = self.clip.tokenize(negative if negative else "")
                
                # Encode with pooled output
                cond, pooled = self.clip.encode_from_tokens(tokens_pos, return_pooled=True)
                uncond, pooled_neg = self.clip.encode_from_tokens(tokens_neg, return_pooled=True)
                
                # Build SDXL conditioning format
                positive_out = [[
                    cond.cpu(),
                    {
                        "pooled_output": pooled.cpu(),
                        "width": width,
                        "height": height,
                        "crop_w": crop_w,
                        "crop_h": crop_h,
                        "target_width": target_width,
                        "target_height": target_height
                    }
                ]]
                
                negative_out = [[
                    uncond.cpu(),
                    {
                        "pooled_output": pooled_neg.cpu(),
                        "width": width,
                        "height": height,
                        "crop_w": crop_w,
                        "crop_h": crop_h,
                        "target_width": target_width,
                        "target_height": target_height
                    }
                ]]
                
                return (positive_out, negative_out)
            except Exception as e:
                logger.error(f"CLIP SDXL encode error: {e}")
                raise
    
    def get_info(self) -> Dict[str, Any]:
        """Get daemon status info"""
        info = {
            "status": "ok",
            "device": self.device,
            "request_count": self.request_count,
            "uptime_seconds": time.time() - self.start_time if self.start_time else 0,
            "vae_loaded": self.vae is not None,
            "clip_loaded": self.clip is not None,
            "current_vae": getattr(self, 'current_vae_path', None),
            "current_clip_l": getattr(self, 'current_clip_l_path', None),
            "current_clip_g": getattr(self, 'current_clip_g_path', None),
            "current_t5xxl": getattr(self, 'current_t5xxl_path', None),
        }
        
        if 'cuda' in self.device:
            device_idx = int(self.device.split(':')[1]) if ':' in self.device else 0
            info["vram_used_gb"] = torch.cuda.memory_allocated(device_idx) / 1024**3
            info["vram_total_gb"] = torch.cuda.get_device_properties(device_idx).total_memory / 1024**3
        
        return info
    
    def configure(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dynamically reconfigure the daemon with new models.
        
        This allows workflows to specify which VAE/CLIP to use at runtime,
        instead of requiring hardcoded paths in config.py.
        
        Args:
            config: Dict with optional keys:
                - device: GPU device to use
                - vae_path: Path to VAE model
                - clip_l_path: Path to CLIP-L model
                - clip_g_path: Path to CLIP-G model
                - t5xxl_path: Path to T5-XXL model (for Flux)
                - embeddings_dir: Path to embeddings directory
        
        Returns:
            Dict with status and list of reloaded models
        """
        import comfy.sd
        import comfy.utils
        
        reloaded = []
        
        with self.lock:
            try:
                # Check if device change is requested
                new_device = config.get("device")
                if new_device and new_device != self.device:
                    logger.info(f"Device change requested: {self.device} -> {new_device}")
                    self.device = new_device
                    # Note: Full device migration would require reloading all models
                
                # Check if VAE needs reload
                new_vae_path = config.get("vae_path")
                if new_vae_path and new_vae_path != getattr(self, 'current_vae_path', None):
                    if os.path.exists(new_vae_path):
                        logger.info(f"Loading new VAE: {new_vae_path}")
                        sd = comfy.utils.load_torch_file(new_vae_path)
                        self.vae = comfy.sd.VAE(sd=sd)
                        self.current_vae_path = new_vae_path
                        reloaded.append("vae")
                        logger.info("VAE reloaded successfully")
                    else:
                        return {"status": "error", "error": f"VAE not found: {new_vae_path}"}
                
                # Check if CLIP needs reload
                new_clip_l = config.get("clip_l_path")
                new_clip_g = config.get("clip_g_path")
                new_t5xxl = config.get("t5xxl_path")
                embeddings_dir = config.get("embeddings_dir", EMBEDDINGS_DIR)
                
                # Determine if CLIP reload needed
                clip_changed = False
                if new_clip_l and new_clip_l != getattr(self, 'current_clip_l_path', None):
                    clip_changed = True
                if new_clip_g and new_clip_g != getattr(self, 'current_clip_g_path', None):
                    clip_changed = True
                if new_t5xxl and new_t5xxl != getattr(self, 'current_t5xxl_path', None):
                    clip_changed = True
                
                if clip_changed:
                    # Build clip paths list
                    clip_paths = []
                    if new_clip_l and os.path.exists(new_clip_l):
                        clip_paths.append(new_clip_l)
                        self.current_clip_l_path = new_clip_l
                    elif getattr(self, 'current_clip_l_path', None):
                        clip_paths.append(self.current_clip_l_path)
                    
                    if new_clip_g and os.path.exists(new_clip_g):
                        clip_paths.append(new_clip_g)
                        self.current_clip_g_path = new_clip_g
                    elif getattr(self, 'current_clip_g_path', None):
                        clip_paths.append(self.current_clip_g_path)
                    
                    if new_t5xxl and os.path.exists(new_t5xxl):
                        clip_paths.append(new_t5xxl)
                        self.current_t5xxl_path = new_t5xxl
                    elif getattr(self, 'current_t5xxl_path', None):
                        clip_paths.append(self.current_t5xxl_path)
                    
                    if clip_paths:
                        logger.info(f"Loading new CLIP: {clip_paths}")
                        emb_dir = embeddings_dir if os.path.exists(embeddings_dir) else None
                        self.clip = comfy.sd.load_clip(
                            ckpt_paths=clip_paths,
                            embedding_directory=emb_dir
                        )
                        reloaded.append("clip")
                        logger.info("CLIP reloaded successfully")
                
                # Clear CUDA cache after model swap
                torch.cuda.empty_cache()
                
                return {"status": "ok", "reloaded": reloaded}
                
            except Exception as e:
                logger.error(f"Configure error: {e}")
                return {"status": "error", "error": str(e)}
    
    def handle_request(self, conn: socket.socket, addr: Tuple[str, int]):
        """Handle incoming request from ComfyUI node"""
        try:
            # Receive data with end marker
            data = b""
            while True:
                chunk = conn.recv(1048576)  # 1MB chunks
                if not chunk:
                    break
                data += chunk
                if b"<<END>>" in data:
                    data = data.replace(b"<<END>>", b"")
                    break
            
            if not data:
                return
            
            request = pickle.loads(data)
            cmd = request.get("cmd", "unknown")
            
            self.request_count += 1
            logger.debug(f"Request #{self.request_count}: {cmd}")
            
            # Route command
            if cmd == "health":
                result = {"status": "ok"}
            elif cmd == "info":
                result = self.get_info()
            elif cmd == "configure":
                result = self.configure(request.get("config", {}))
            elif cmd == "vae_encode":
                result = self.vae_encode(request["pixels"])
            elif cmd == "vae_decode":
                result = self.vae_decode(request["latents"])
            elif cmd == "clip_encode":
                result = self.clip_encode(
                    request["positive"],
                    request.get("negative", "")
                )
            elif cmd == "clip_encode_sdxl":
                result = self.clip_encode_sdxl(
                    request["positive"],
                    request.get("negative", ""),
                    request.get("width", 1024),
                    request.get("height", 1024),
                    request.get("crop_w", 0),
                    request.get("crop_h", 0),
                    request.get("target_width", 1024),
                    request.get("target_height", 1024)
                )
            else:
                result = {"error": f"Unknown command: {cmd}"}
            
            # Send response
            response = pickle.dumps(result)
            conn.sendall(response)
            
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            try:
                conn.sendall(pickle.dumps({"error": str(e)}))
            except:
                pass
        finally:
            conn.close()
    
    def run(self):
        """Main server loop"""
        # Load models first
        self.load_models()
        
        # Create server socket
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server.bind((DAEMON_HOST, DAEMON_PORT))
            server.listen(MAX_WORKERS)
            logger.info(f"Listening on {DAEMON_HOST}:{DAEMON_PORT}")
            logger.info("Ready to accept connections from ComfyUI instances")
            logger.info("Press Ctrl+C to stop")
            
            while True:
                conn, addr = server.accept()
                thread = threading.Thread(
                    target=self.handle_request,
                    args=(conn, addr),
                    daemon=True
                )
                thread.start()
                
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            server.close()


def main():
    """Entry point"""
    print("=" * 60)
    print("  Luna VAE/CLIP Daemon")
    print("  Shared model server for multi-instance ComfyUI")
    print("=" * 60)
    print()
    
    daemon = VAECLIPDaemon()
    daemon.run()


if __name__ == "__main__":
    main()
