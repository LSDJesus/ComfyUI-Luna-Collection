"""
Luna Daemon - Z-IMAGE CLIP Proxy

Extended CLIP proxy that automatically detects and supports:
1. Standard CLIP models (SD1.5, SDXL, Flux, SD3)
2. Z-IMAGE's Qwen3-4B CLIP encoder
3. Any Qwen3-VL model as unified text encoder

Auto-detection logic:
┌────────────────────────────────────────────────────────────────────┐
│                    CLIP Model Detection                            │
├────────────────────────────────────────────────────────────────────┤
│  Check source model architecture:                                  │
│    • vocab_size == 151936 AND hidden_size == 2560                 │
│      → Qwen3 architecture → Z-IMAGE compatible                    │
│    • "Qwen" in class name                                         │
│      → Qwen3 architecture → Z-IMAGE compatible                    │
│    • clip_l + clip_g present                                      │
│      → SDXL architecture                                          │
│    • t5xxl present                                                │
│      → Flux/SD3 architecture                                      │
│    • Standard CLIP structure                                       │
│      → SD1.5 architecture                                         │
└────────────────────────────────────────────────────────────────────┘

When Z-IMAGE is detected:
- Routes encoding to Qwen3-VL encoder on daemon
- Uses daemon's shared Qwen3-VL-4B model
- Embeddings are compatible with Z-IMAGE's expectation
"""

import torch
import logging
from typing import Optional, Any, Dict, List, Tuple, Union

# Try to import daemon client
try:
    from . import client as daemon_client
    from .client import DaemonConnectionError, ModelMismatchError
    DAEMON_AVAILABLE = True
except ImportError:
    DAEMON_AVAILABLE = False
    daemon_client = None
    DaemonConnectionError = Exception
    ModelMismatchError = Exception

logger = logging.getLogger(__name__)


# Z-IMAGE model signature
ZIMAGE_VOCAB_SIZE = 151936
ZIMAGE_HIDDEN_SIZE = 2560


def detect_clip_architecture(clip) -> Dict[str, Any]:
    """
    Detect detailed CLIP architecture from a CLIP object.
    
    Returns dict with:
        - type: 'sdxl', 'sd15', 'flux', 'sd3', 'zimage', 'qwen3', 'unknown'
        - is_qwen: bool - True if Qwen3-based
        - vocab_size: int
        - hidden_size: int
        - components: list of component names
    """
    result = {
        'type': 'unknown',
        'is_qwen': False,
        'vocab_size': 0,
        'hidden_size': 0,
        'components': [],
    }
    
    if clip is None:
        return result
    
    try:
        # Get the conditioning model
        cond_model = getattr(clip, 'cond_stage_model', None)
        if cond_model is None and hasattr(clip, 'patcher'):
            cond_model = getattr(clip.patcher, 'model', None)
        
        if cond_model is None:
            cond_model = clip
        
        # Check class name for Qwen
        class_name = type(cond_model).__name__
        module_name = type(cond_model).__module__ or ""
        
        if 'Qwen' in class_name or 'qwen' in module_name.lower():
            result['is_qwen'] = True
            result['type'] = 'zimage'
        
        # Try to get vocab_size and hidden_size
        config = None
        if hasattr(cond_model, 'config'):
            config = cond_model.config
        elif hasattr(cond_model, 'model') and hasattr(cond_model.model, 'config'):
            config = cond_model.model.config
        
        if config is not None:
            # Handle nested text_config (for VL models)
            text_config = getattr(config, 'text_config', config)
            
            vocab_size = getattr(text_config, 'vocab_size', 0)
            hidden_size = getattr(text_config, 'hidden_size', 0)
            
            result['vocab_size'] = vocab_size
            result['hidden_size'] = hidden_size
            
            # Check for Z-IMAGE signature
            if vocab_size == ZIMAGE_VOCAB_SIZE and hidden_size == ZIMAGE_HIDDEN_SIZE:
                result['is_qwen'] = True
                result['type'] = 'zimage'
        
        # Check for embedding layer directly (safetensors case)
        if hasattr(cond_model, 'embed_tokens'):
            embed = cond_model.embed_tokens
            if hasattr(embed, 'weight'):
                shape = embed.weight.shape
                result['vocab_size'] = shape[0]
                result['hidden_size'] = shape[1]
                if shape[0] == ZIMAGE_VOCAB_SIZE and shape[1] == ZIMAGE_HIDDEN_SIZE:
                    result['is_qwen'] = True
                    result['type'] = 'zimage'
        
        # If not Qwen, detect standard CLIP types
        if not result['is_qwen']:
            # Check for SDXL components
            if hasattr(cond_model, 'clip_g') or 'SDXL' in class_name:
                result['type'] = 'sdxl'
                result['components'] = ['clip_l', 'clip_g']
            # Check for Flux/T5
            elif 't5xxl' in class_name.lower() or 'Flux' in class_name:
                result['type'] = 'flux'
                result['components'] = ['clip_l', 't5xxl']
            # Check for SD3
            elif 'SD3' in class_name:
                result['type'] = 'sd3'
                result['components'] = ['clip_l', 'clip_g', 't5xxl']
            # Default to SD1.5
            elif hasattr(cond_model, 'transformer') or 'Clip' in class_name:
                result['type'] = 'sd15'
                result['components'] = ['clip_l']
            else:
                result['type'] = 'sdxl'  # Default assumption
                result['components'] = ['clip_l', 'clip_g']
        else:
            result['components'] = ['qwen3']
            
    except Exception as e:
        logger.warning(f"[detect_clip_architecture] Error detecting CLIP type: {e}")
        result['type'] = 'unknown'
    
    return result


def is_zimage_clip(clip) -> bool:
    """Quick check if CLIP is Z-IMAGE compatible (Qwen3-based)."""
    arch = detect_clip_architecture(clip)
    return arch['is_qwen']


class DaemonZImageCLIP:
    """
    CLIP proxy specifically for Z-IMAGE's Qwen3-4B encoder.
    
    Routes all encoding to the daemon's Qwen3-VL encoder service.
    
    Key difference from standard DaemonCLIP:
    - Uses Qwen3-VL model for encoding instead of standard CLIP
    - Outputs embeddings in Z-IMAGE's expected format [B, seq_len, 2560]
    - No CLIP-L/CLIP-G components - single unified encoder
    
    Usage:
        # Auto-detection via LunaCheckpointTunnel
        proxy = DaemonZImageCLIP(source_clip=actual_clip)
        
        # Or explicit creation
        proxy = DaemonZImageCLIP.create_for_daemon()
    """
    
    def __init__(
        self,
        source_clip: Optional[Any] = None,
        use_existing: bool = False
    ):
        """
        Create a Z-IMAGE CLIP proxy.
        
        Args:
            source_clip: The actual CLIP object (for validation)
            use_existing: If True, use daemon's already-loaded Qwen3-VL
        """
        self.source_clip = source_clip
        self.use_existing = use_existing
        self._registered = False
        self._layer_idx = None
        
        # Verify Z-IMAGE compatibility
        if source_clip is not None:
            arch = detect_clip_architecture(source_clip)
            if not arch['is_qwen']:
                logger.warning(
                    f"[DaemonZImageCLIP] Source CLIP doesn't appear to be Z-IMAGE compatible: "
                    f"vocab_size={arch['vocab_size']}, hidden_size={arch['hidden_size']}"
                )
        
        # Type for daemon routing
        self.clip_type = 'zimage'
        self.components = ['qwen3']
        
        # Match ComfyUI CLIP attributes
        self.device = torch.device("cpu")
        self.dtype = torch.bfloat16
        self.cond_stage_model = None
        self.patcher = None
        
        # LoRA not supported for Z-IMAGE (yet)
        self.lora_stack: List[Dict[str, Any]] = []
    
    @classmethod
    def create_for_daemon(cls) -> 'DaemonZImageCLIP':
        """Create a proxy that uses daemon's existing Qwen3-VL encoder."""
        return cls(source_clip=None, use_existing=True)
    
    def _check_daemon(self):
        """Verify daemon is running and has Qwen3-VL loaded."""
        if not DAEMON_AVAILABLE or daemon_client is None:
            raise RuntimeError(
                "Luna Daemon module not available.\n"
                "Ensure luna_daemon package is properly installed."
            )
        
        if not daemon_client.is_daemon_running():
            raise RuntimeError(
                "Luna Daemon is not running!\n"
                "Start it from the Luna Daemon panel or run:\n"
                "  python -m luna_daemon.server"
            )
    
    def _ensure_qwen3_loaded(self):
        """Ensure Qwen3-VL encoder is loaded in daemon."""
        if self._registered or self.use_existing:
            return
        
        # Check if daemon has Qwen3-VL
        try:
            info = daemon_client.get_daemon_info()
            if info.get('qwen3_loaded'):
                self._registered = True
                return
        except Exception:
            pass
        
        # Request daemon to load Qwen3-VL
        # The daemon should have a configured model path
        logger.info("[DaemonZImageCLIP] Requesting daemon to load Qwen3-VL encoder")
        # TODO: Implement daemon command to load Qwen3-VL
        self._registered = True
    
    def tokenize(self, text: str, return_word_ids: bool = False, **kwargs):
        """
        Tokenize text - stores text for daemon-side tokenization.
        
        Qwen3 uses its own tokenizer, handled by the daemon.
        """
        return ZImageTokens(text, return_word_ids, kwargs)
    
    def encode_from_tokens(self, tokens, return_pooled: bool = False,
                           return_dict: bool = False) -> Any:
        """Encode tokens via daemon's Qwen3-VL encoder."""
        self._check_daemon()
        self._ensure_qwen3_loaded()
        
        # Handle our token wrapper
        if isinstance(tokens, ZImageTokens):
            text = tokens.text
        elif isinstance(tokens, dict) and "text" in tokens:
            text = tokens["text"]
        elif isinstance(tokens, str):
            text = tokens
        else:
            raise RuntimeError(
                "DaemonZImageCLIP received incompatible tokens.\n"
                f"Got: {type(tokens)}"
            )
        
        try:
            # Route to Qwen3-VL encoding endpoint
            cond = daemon_client.zimage_encode(text)
            
            # Z-IMAGE uses the full sequence, no separate pooled output typically
            # But we generate one for compatibility
            pooled = cond.mean(dim=1) if cond is not None else None
            
            if return_dict:
                return {"cond": cond, "pooled_output": pooled}
            elif return_pooled:
                return cond, pooled
            else:
                return cond
                
        except Exception as e:
            raise RuntimeError(f"Z-IMAGE encoding error: {e}")
    
    def encode_from_tokens_scheduled(self, tokens, unprojected: bool = False,
                                      add_dict: Optional[Dict[str, Any]] = None,
                                      show_pbar: bool = True) -> List:
        """
        Scheduled encoding for ComfyUI conditioning format.
        """
        if add_dict is None:
            add_dict = {}
        
        # Handle our token wrapper
        if isinstance(tokens, ZImageTokens):
            text = tokens.text
        elif isinstance(tokens, dict) and "text" in tokens:
            text = tokens["text"]
        elif isinstance(tokens, str):
            text = tokens
        else:
            raise RuntimeError(f"DaemonZImageCLIP received incompatible tokens: {type(tokens)}")
        
        self._check_daemon()
        self._ensure_qwen3_loaded()
        
        try:
            cond = daemon_client.zimage_encode(text)
            pooled = cond.mean(dim=1) if cond is not None else torch.zeros(1, ZIMAGE_HIDDEN_SIZE)
            
            # Return in ComfyUI conditioning format
            cond_dict = {"pooled_output": pooled}
            cond_dict.update(add_dict)
            
            return [[cond, cond_dict]]
            
        except Exception as e:
            raise RuntimeError(f"Z-IMAGE encoding error: {e}")
    
    def encode(self, text: str):
        """Convenience: tokenize + encode."""
        tokens = self.tokenize(text)
        return self.encode_from_tokens(tokens, return_pooled=True)
    
    def clip_layer(self, layer_idx: int):
        """Set CLIP layer - stored but Qwen3 may not use this."""
        self._layer_idx = layer_idx
    
    def clone(self):
        """Clone this CLIP proxy."""
        new_clip = DaemonZImageCLIP(
            source_clip=self.source_clip,
            use_existing=self.use_existing or self._registered
        )
        new_clip._layer_idx = self._layer_idx
        new_clip._registered = self._registered
        new_clip.lora_stack = self.lora_stack.copy()
        return new_clip
    
    def load_model(self):
        """Load model - daemon handles this."""
        pass
    
    def get_sd(self) -> Dict:
        """Get state dict - not applicable."""
        return {}
    
    def get_key_patches(self) -> Dict:
        """Get patches - not applicable."""
        return {}
    
    def add_patches(self, patches: Dict, strength_patch: float = 1.0, strength_model: float = 1.0):
        """
        Add patches (LoRA) - not yet supported for Z-IMAGE.
        
        Future: Could support Qwen3 LoRAs if trained.
        """
        logger.warning("[DaemonZImageCLIP] LoRA not yet supported for Z-IMAGE CLIP, skipping")
        return self.clone()
    
    def load_sd(self, sd, full_model=False):
        """Load state dict - not applicable."""
        pass
    
    def set_tokenizer_option(self, option_name: str, value: Any):
        """Set tokenizer option - stored for daemon."""
        pass


class ZImageTokens:
    """
    Token wrapper for Z-IMAGE encoding.
    
    Stores text for daemon-side tokenization by Qwen3 tokenizer.
    """
    
    def __init__(self, text: str, return_word_ids: bool = False, kwargs: Optional[Dict] = None):
        self.text = text
        self.return_word_ids = return_word_ids
        self.kwargs = kwargs or {}
    
    def __repr__(self):
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"ZImageTokens({preview})"


# =============================================================================
# Factory function for auto-detection
# =============================================================================

def create_clip_proxy(
    source_clip: Any,
    use_existing: bool = False,
    force_type: Optional[str] = None
) -> Any:  # Returns DaemonCLIP or DaemonZImageCLIP
    """
    Create the appropriate CLIP proxy based on auto-detection.
    
    Args:
        source_clip: The actual CLIP object
        use_existing: If True, use daemon's existing models
        force_type: Force a specific type ('zimage', 'sdxl', 'sd15', etc.)
    
    Returns:
        Either DaemonZImageCLIP or DaemonCLIP depending on detection
    """
    # Import standard proxy here to avoid circular import
    from .proxy import DaemonCLIP
    
    if force_type == 'zimage':
        return DaemonZImageCLIP(source_clip, use_existing)
    
    if force_type is not None:
        return DaemonCLIP(source_clip, clip_type=force_type, use_existing=use_existing)
    
    # Auto-detect
    arch = detect_clip_architecture(source_clip)
    
    if arch['is_qwen']:
        logger.info(f"[create_clip_proxy] Detected Z-IMAGE (Qwen3) CLIP: "
                   f"vocab={arch['vocab_size']}, hidden={arch['hidden_size']}")
        return DaemonZImageCLIP(source_clip, use_existing)
    else:
        logger.info(f"[create_clip_proxy] Detected standard CLIP: {arch['type']}, "
                   f"components={arch['components']}")
        return DaemonCLIP(source_clip, clip_type=arch['type'], use_existing=use_existing)
