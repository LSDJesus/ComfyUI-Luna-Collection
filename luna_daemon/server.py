"""
Luna VAE/CLIP Daemon Server
Loads CLIP and VAE models on-demand from first workflow request,
then validates all subsequent requests use the same models.

Usage:
    python -m custom_nodes.ComfyUI-Luna-Collection.luna_daemon.server
    
Or from ComfyUI root:
    python -m luna_daemon.server
"""

import os
import sys
import socket
import pickle
import struct
import threading
import time
import logging
from typing import Any, Dict, Tuple, Optional

import torch

# Add ComfyUI to path if needed
comfy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if comfy_path not in sys.path:
    sys.path.insert(0, comfy_path)

from .config import (
    DAEMON_HOST, DAEMON_PORT, SHARED_DEVICE,
    EMBEDDINGS_DIR, MAX_WORKERS, LOG_LEVEL
)

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='[%(asctime)s] [Daemon] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class ModelMismatchError(Exception):
    """Raised when workflow tries to use a different model than what's loaded"""
    pass


class VAECLIPDaemon:
    """
    Daemon server that loads VAE and CLIP models on-demand from first request,
    then validates all subsequent requests use the same models.
    """
    
    def __init__(self, device: str = SHARED_DEVICE):
        self.device = device
        self.lock = threading.Lock()
        self.request_count = 0
        self.start_time = time.time()
        
        # Models - loaded on-demand from first request
        self.vae = None
        self.clip = None
        
        # Track which models are loaded (by path/name)
        self.loaded_vae_path: Optional[str] = None
        self.loaded_clip_paths: Optional[Tuple[str, ...]] = None
        
        # Loading state
        self.vae_loading = False
        self.clip_loading = False
    
    def _normalize_path(self, path: str) -> str:
        """Normalize path for comparison"""
        return os.path.normpath(os.path.abspath(path)).lower()
    
    def _paths_match(self, path1: str, path2: str) -> bool:
        """Check if two paths refer to the same file"""
        return self._normalize_path(path1) == self._normalize_path(path2)
    
    def _clip_paths_match(self, paths1: Tuple[str, ...], paths2: Tuple[str, ...]) -> bool:
        """Check if two sets of CLIP paths match"""
        norm1 = tuple(sorted(self._normalize_path(p) for p in paths1))
        norm2 = tuple(sorted(self._normalize_path(p) for p in paths2))
        return norm1 == norm2
    
    def _log_vram(self):
        """Log VRAM usage"""
        if 'cuda' in self.device:
            device_idx = int(self.device.split(':')[1]) if ':' in self.device else 0
            vram_used = torch.cuda.memory_allocated(device_idx) / 1024**3
            vram_total = torch.cuda.get_device_properties(device_idx).total_memory / 1024**3
            logger.info(f"VRAM: {vram_used:.2f} / {vram_total:.2f} GB")
    
    def load_vae(self, vae_path: str) -> None:
        """Load VAE on-demand from first request"""
        with self.lock:
            # Already loaded?
            if self.vae is not None:
                if self._paths_match(vae_path, self.loaded_vae_path):
                    return  # Same VAE, already loaded
                else:
                    raise ModelMismatchError(
                        f"VAE mismatch! Daemon has loaded:\n"
                        f"  {os.path.basename(self.loaded_vae_path)}\n"
                        f"but workflow requested:\n"
                        f"  {os.path.basename(vae_path)}\n"
                        f"Restart daemon to use a different VAE."
                    )
            
            # Prevent concurrent loading
            if self.vae_loading:
                # Wait for loading to complete
                while self.vae_loading:
                    time.sleep(0.1)
                return
            
            self.vae_loading = True
        
        try:
            logger.info(f"Loading VAE on-demand: {vae_path}")
            
            if not os.path.exists(vae_path):
                raise FileNotFoundError(f"VAE file not found: {vae_path}")
            
            import comfy.sd
            import comfy.utils
            
            sd = comfy.utils.load_torch_file(vae_path)
            self.vae = comfy.sd.VAE(sd=sd)
            self.loaded_vae_path = vae_path
            
            logger.info(f"VAE loaded: {os.path.basename(vae_path)}")
            self._log_vram()
            
        finally:
            self.vae_loading = False
    
    def load_clip(self, clip_paths: Tuple[str, ...]) -> None:
        """Load CLIP on-demand from first request"""
        with self.lock:
            # Already loaded?
            if self.clip is not None:
                if self._clip_paths_match(clip_paths, self.loaded_clip_paths):
                    return  # Same CLIP, already loaded
                else:
                    loaded_names = ", ".join(os.path.basename(p) for p in self.loaded_clip_paths)
                    requested_names = ", ".join(os.path.basename(p) for p in clip_paths)
                    raise ModelMismatchError(
                        f"CLIP mismatch! Daemon has loaded:\n"
                        f"  {loaded_names}\n"
                        f"but workflow requested:\n"
                        f"  {requested_names}\n"
                        f"Restart daemon to use different CLIP models."
                    )
            
            # Prevent concurrent loading
            if self.clip_loading:
                while self.clip_loading:
                    time.sleep(0.1)
                return
            
            self.clip_loading = True
        
        try:
            logger.info(f"Loading CLIP on-demand: {clip_paths}")
            
            # Validate paths exist
            valid_paths = []
            for p in clip_paths:
                if os.path.exists(p):
                    valid_paths.append(p)
                else:
                    logger.warning(f"CLIP file not found: {p}")
            
            if not valid_paths:
                raise FileNotFoundError(f"No valid CLIP files found in: {clip_paths}")
            
            import comfy.sd
            
            embeddings_dir = EMBEDDINGS_DIR if os.path.exists(EMBEDDINGS_DIR) else None
            self.clip = comfy.sd.load_clip(
                ckpt_paths=valid_paths,
                embedding_directory=embeddings_dir
            )
            self.loaded_clip_paths = tuple(valid_paths)
            
            clip_names = ", ".join(os.path.basename(p) for p in valid_paths)
            logger.info(f"CLIP loaded: {clip_names}")
            self._log_vram()
            
        finally:
            self.clip_loading = False
    
    def unload_models(self) -> Dict[str, Any]:
        """Unload all models to allow loading different ones"""
        with self.lock:
            old_vae = self.loaded_vae_path
            old_clip = self.loaded_clip_paths
            
            self.vae = None
            self.clip = None
            self.loaded_vae_path = None
            self.loaded_clip_paths = None
            
            torch.cuda.empty_cache()
            
            logger.info("All models unloaded")
            return {
                "status": "ok",
                "unloaded_vae": old_vae,
                "unloaded_clip": list(old_clip) if old_clip else None
            }
    
    def vae_encode(self, pixels: torch.Tensor, vae_path: str) -> torch.Tensor:
        """Encode image pixels to latent space"""
        # Load or validate VAE
        self.load_vae(vae_path)
        
        with self.lock:
            try:
                if pixels.dim() == 3:
                    pixels = pixels.unsqueeze(0)
                latents = self.vae.encode(pixels)
                return latents.cpu()
            except Exception as e:
                logger.error(f"VAE encode error: {e}")
                raise
    
    def vae_decode(self, latents: torch.Tensor, vae_path: str) -> torch.Tensor:
        """Decode latent space to image pixels"""
        # Load or validate VAE
        self.load_vae(vae_path)
        
        with self.lock:
            try:
                pixels = self.vae.decode(latents)
                return pixels.cpu()
            except Exception as e:
                logger.error(f"VAE decode error: {e}")
                raise
    
    def clip_encode(self, positive: str, negative: str, clip_paths: Tuple[str, ...]) -> Tuple:
        """Encode text to CLIP conditioning"""
        # Load or validate CLIP
        self.load_clip(clip_paths)
        
        with self.lock:
            try:
                tokens_pos = self.clip.tokenize(positive)
                cond, pooled = self.clip.encode_from_tokens(tokens_pos, return_pooled=True)
                
                tokens_neg = self.clip.tokenize(negative if negative else "")
                uncond, pooled_neg = self.clip.encode_from_tokens(tokens_neg, return_pooled=True)
                
                return (cond.cpu(), pooled.cpu(), uncond.cpu(), pooled_neg.cpu())
            except Exception as e:
                logger.error(f"CLIP encode error: {e}")
                raise
    
    def clip_encode_sdxl(
        self,
        positive: str,
        negative: str,
        clip_paths: Tuple[str, ...],
        width: int = 1024,
        height: int = 1024,
        crop_w: int = 0,
        crop_h: int = 0,
        target_width: int = 1024,
        target_height: int = 1024
    ) -> Tuple[list, list]:
        """Encode text with SDXL-specific size conditioning"""
        # Load or validate CLIP
        self.load_clip(clip_paths)
        
        with self.lock:
            try:
                tokens_pos = self.clip.tokenize(positive)
                tokens_neg = self.clip.tokenize(negative if negative else "")
                
                cond, pooled = self.clip.encode_from_tokens(tokens_pos, return_pooled=True)
                uncond, pooled_neg = self.clip.encode_from_tokens(tokens_neg, return_pooled=True)
                
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
            "uptime_seconds": time.time() - self.start_time,
            "vae_loaded": self.vae is not None,
            "clip_loaded": self.clip is not None,
            "loaded_vae": os.path.basename(self.loaded_vae_path) if self.loaded_vae_path else None,
            "loaded_vae_path": self.loaded_vae_path,
            "loaded_clip": [os.path.basename(p) for p in self.loaded_clip_paths] if self.loaded_clip_paths else None,
            "loaded_clip_paths": list(self.loaded_clip_paths) if self.loaded_clip_paths else None,
        }
        
        if 'cuda' in self.device:
            device_idx = int(self.device.split(':')[1]) if ':' in self.device else 0
            info["vram_used_gb"] = round(torch.cuda.memory_allocated(device_idx) / 1024**3, 2)
            info["vram_total_gb"] = round(torch.cuda.get_device_properties(device_idx).total_memory / 1024**3, 2)
        
        return info
    
    def handle_request(self, conn: socket.socket, addr: Tuple[str, int]):
        """Handle incoming request from ComfyUI node using Length-Prefix Protocol"""
        try:
            # Read 4-byte length header
            header = b""
            while len(header) < 4:
                chunk = conn.recv(4 - len(header))
                if not chunk:
                    return  # Connection closed
                header += chunk
            
            request_len = struct.unpack('>I', header)[0]
            
            # Read exact payload
            data = b""
            while len(data) < request_len:
                chunk_size = min(request_len - len(data), 1048576)  # 1MB chunks
                chunk = conn.recv(chunk_size)
                if not chunk:
                    return  # Connection closed
                data += chunk
            
            request = pickle.loads(data)
            cmd = request.get("cmd", "unknown")
            
            self.request_count += 1
            logger.debug(f"Request #{self.request_count}: {cmd}")
            
            # Route command
            if cmd == "health":
                result = {"status": "ok"}
            
            elif cmd == "info":
                result = self.get_info()
            
            elif cmd == "unload":
                result = self.unload_models()
            
            elif cmd == "shutdown":
                logger.info("Shutdown requested")
                result = {"status": "ok", "message": "Shutting down"}
                # Send response before shutdown (with length prefix)
                response = pickle.dumps(result)
                conn.sendall(struct.pack('>I', len(response)) + response)
                conn.close()
                # Exit the process
                import sys
                sys.exit(0)
            
            elif cmd == "vae_encode":
                result = self.vae_encode(
                    request["pixels"],
                    request["vae_path"]
                )
            
            elif cmd == "vae_decode":
                result = self.vae_decode(
                    request["latents"],
                    request["vae_path"]
                )
            
            elif cmd == "clip_encode":
                result = self.clip_encode(
                    request["positive"],
                    request.get("negative", ""),
                    tuple(request["clip_paths"])
                )
            
            elif cmd == "clip_encode_sdxl":
                result = self.clip_encode_sdxl(
                    request["positive"],
                    request.get("negative", ""),
                    tuple(request["clip_paths"]),
                    request.get("width", 1024),
                    request.get("height", 1024),
                    request.get("crop_w", 0),
                    request.get("crop_h", 0),
                    request.get("target_width", 1024),
                    request.get("target_height", 1024)
                )
            
            else:
                result = {"error": f"Unknown command: {cmd}"}
            
            # Send response with length prefix
            response = pickle.dumps(result)
            conn.sendall(struct.pack('>I', len(response)) + response)
            
        except ModelMismatchError as e:
            logger.warning(f"Model mismatch: {e}")
            try:
                error_response = pickle.dumps({"error": str(e), "type": "model_mismatch"})
                conn.sendall(struct.pack('>I', len(error_response)) + error_response)
            except:
                pass
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            try:
                error_response = pickle.dumps({"error": str(e)})
                conn.sendall(struct.pack('>I', len(error_response)) + error_response)
            except:
                pass
        finally:
            conn.close()
    
    def run(self, host: str = DAEMON_HOST, port: int = DAEMON_PORT):
        """Main server loop - starts without loading models (on-demand)"""
        # Create server socket
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server.bind((host, port))
            server.listen(MAX_WORKERS)
            
            logger.info(f"Daemon ready on {host}:{port}")
            logger.info(f"Device: {self.device}")
            logger.info("Waiting for first workflow to load models on-demand...")
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Luna VAE/CLIP Daemon Server")
    parser.add_argument(
        "--gpu", "-g",
        type=int,
        default=None,
        help="GPU index to use (e.g., 0, 1, 2). Overrides config.py SHARED_DEVICE"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=DAEMON_PORT,
        help=f"Port to listen on (default: {DAEMON_PORT})"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DAEMON_HOST,
        help=f"Host to bind to (default: {DAEMON_HOST})"
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.gpu is not None:
        device = f"cuda:{args.gpu}"
    else:
        device = SHARED_DEVICE
    
    print("=" * 60)
    print("  Luna VAE/CLIP Daemon")
    print("  On-demand model loading for multi-instance ComfyUI")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Host:   {args.host}:{args.port}")
    print("  Models: Loaded on first workflow request")
    print("=" * 60)
    print()
    
    daemon = VAECLIPDaemon(device=device)
    daemon.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
