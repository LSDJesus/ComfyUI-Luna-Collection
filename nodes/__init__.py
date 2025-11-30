# Import from subdirectories
from . import preprocessing
from . import performance
from . import upscaling
from . import detailing
from . import loaders
from . import promptcraft

# Import individual files
from .luna_load_parameters import LunaLoadParameters
from .luna_load_preprocessed import (
    LunaSelectPromptFolder,
    LunaLoadPreprocessedPrompt,
    LunaModifyPreprocessedPrompt,
    LunaEmbeddingCache,
    LunaOptimizedPreprocessedLoader,
    LunaCacheManager,
    LunaPerformanceMonitor,
    LunaWildcardPromptGenerator,
    LunaListPreprocessedPrompts,
    LunaSaveNegativePrompt,
    LunaSinglePromptProcessor
)
from .luna_multi_saver import LunaMultiSaver
from .luna_parameters_bridge import LunaParametersBridge
from .luna_sampler import LunaSampler
from .luna_yaml_wildcard import (
    LunaYAMLWildcard,
    LunaYAMLWildcardBatch,
    LunaYAMLWildcardExplorer,
    LunaWildcardBuilder,
    LunaLoRARandomizer,
    LunaYAMLInjector,
    LunaYAMLPathExplorer,
)

# Import shared daemon nodes
from .luna_shared_vae import (
    LunaSharedVAEEncode,
    LunaSharedVAEDecode,
    LunaDaemonStatus,
)
from .luna_shared_clip import (
    LunaSharedCLIPEncode,
    LunaSharedCLIPEncodeSDXL,
    LunaSharedCLIPEncodeDual,
)

# Combine all node mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Add mappings from subdirectories
NODE_CLASS_MAPPINGS.update(preprocessing.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(preprocessing.NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(performance.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(performance.NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(upscaling.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(upscaling.NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(detailing.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(detailing.NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(loaders.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(loaders.NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(promptcraft.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(promptcraft.NODE_DISPLAY_NAME_MAPPINGS)

# Add mappings from individual files
NODE_CLASS_MAPPINGS.update({
    "LunaLoadParameters": LunaLoadParameters,
    "LunaSelectPromptFolder": LunaSelectPromptFolder,
    "LunaLoadPreprocessedPrompt": LunaLoadPreprocessedPrompt,
    "LunaModifyPreprocessedPrompt": LunaModifyPreprocessedPrompt,
    "LunaEmbeddingCache": LunaEmbeddingCache,
    "LunaOptimizedPreprocessedLoader": LunaOptimizedPreprocessedLoader,
    "LunaCacheManager": LunaCacheManager,
    "LunaPerformanceMonitor": LunaPerformanceMonitor,
    "LunaWildcardPromptGenerator": LunaWildcardPromptGenerator,
    "LunaListPreprocessedPrompts": LunaListPreprocessedPrompts,
    "LunaSaveNegativePrompt": LunaSaveNegativePrompt,
    "LunaSinglePromptProcessor": LunaSinglePromptProcessor,
    "LunaMultiSaver": LunaMultiSaver,
    "LunaParametersBridge": LunaParametersBridge,
    "LunaSampler": LunaSampler,
    "LunaYAMLWildcard": LunaYAMLWildcard,
    "LunaYAMLWildcardBatch": LunaYAMLWildcardBatch,
    "LunaYAMLWildcardExplorer": LunaYAMLWildcardExplorer,
    "LunaWildcardBuilder": LunaWildcardBuilder,
    "LunaLoRARandomizer": LunaLoRARandomizer,
    "LunaYAMLInjector": LunaYAMLInjector,
    "LunaYAMLPathExplorer": LunaYAMLPathExplorer,
    # Shared daemon nodes
    "LunaSharedVAEEncode": LunaSharedVAEEncode,
    "LunaSharedVAEDecode": LunaSharedVAEDecode,
    "LunaDaemonStatus": LunaDaemonStatus,
    "LunaSharedCLIPEncode": LunaSharedCLIPEncode,
    "LunaSharedCLIPEncodeSDXL": LunaSharedCLIPEncodeSDXL,
    "LunaSharedCLIPEncodeDual": LunaSharedCLIPEncodeDual,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "LunaLoadParameters": "Luna Load Parameters",
    "LunaSelectPromptFolder": "Luna Select Prompt Folder",
    "LunaLoadPreprocessedPrompt": "Luna Load Preprocessed Prompt",
    "LunaModifyPreprocessedPrompt": "Luna Modify Preprocessed Prompt",
    "LunaEmbeddingCache": "Luna Embedding Cache",
    "LunaOptimizedPreprocessedLoader": "Luna Optimized Preprocessed Loader",
    "LunaCacheManager": "Luna Cache Manager",
    "LunaPerformanceMonitor": "Luna Performance Monitor",
    "LunaWildcardPromptGenerator": "Luna Wildcard Prompt Generator",
    "LunaListPreprocessedPrompts": "Luna List Preprocessed Prompts",
    "LunaSaveNegativePrompt": "Luna Save Negative Prompt",
    "LunaSinglePromptProcessor": "Luna Single Prompt Processor",
    "LunaMultiSaver": "Luna Multi Saver",
    "LunaParametersBridge": "Luna Parameters Bridge",
    "LunaSampler": "Luna Sampler",
    "LunaYAMLWildcard": "Luna YAML Wildcard",
    "LunaYAMLWildcardBatch": "Luna YAML Wildcard Batch",
    "LunaYAMLWildcardExplorer": "Luna YAML Wildcard Explorer",
    "LunaWildcardBuilder": "Luna Wildcard Builder",
    "LunaLoRARandomizer": "Luna LoRA Randomizer",
    "LunaYAMLInjector": "Luna YAML Injector",
    "LunaYAMLPathExplorer": "Luna YAML Path Explorer",
    # Shared daemon nodes
    "LunaSharedVAEEncode": "Luna Shared VAE Encode",
    "LunaSharedVAEDecode": "Luna Shared VAE Decode",
    "LunaDaemonStatus": "Luna Daemon Status",
    "LunaSharedCLIPEncode": "Luna Shared CLIP Encode",
    "LunaSharedCLIPEncodeSDXL": "Luna Shared CLIP Encode (SDXL)",
    "LunaSharedCLIPEncodeDual": "Luna Shared CLIP Encode (Dual)",
})
