# Luna Collection - Node Registration
# Auto-discovers all node files and registers them with ComfyUI

# Import from subdirectories
from . import upscaling
from . import promptcraft

# Import individual node files
from .luna_config_gateway import LunaConfigGateway
from .luna_multi_saver import LunaMultiSaver
from .luna_yaml_wildcard import (
    LunaYAMLWildcard,
    LunaYAMLWildcardBatch,
    LunaYAMLWildcardExplorer,
    LunaWildcardBuilder,
    LunaLoRARandomizer,
    LunaYAMLInjector,
    LunaYAMLPathExplorer,
)
from .luna_wildcard_connections import (
    LunaConnectionMatcher,
    LunaConnectionEditor,
    LunaSmartLoRALinker,
    LunaConnectionStats,
)
from .luna_civitai_scraper import (
    LunaCivitaiScraper,
    LunaCivitaiBatchScraper,
)
from .luna_expression_pack import (
    LunaExpressionPromptBuilder,
    LunaExpressionSlicerSaver,
)

# Import daemon nodes - proxy loaders that return VAE/CLIP objects usable by any node
from .luna_daemon_loader import (
    LunaDaemonVAELoader,
    LunaDaemonCLIPLoader,
)

# Combine all node mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Add mappings from subdirectories
NODE_CLASS_MAPPINGS.update(upscaling.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(upscaling.NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(promptcraft.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(promptcraft.NODE_DISPLAY_NAME_MAPPINGS)

# Add mappings from individual files
NODE_CLASS_MAPPINGS.update({
    # Config
    "LunaConfigGateway": LunaConfigGateway,
    # Image saving
    "LunaMultiSaver": LunaMultiSaver,
    # YAML Wildcards
    "LunaYAMLWildcard": LunaYAMLWildcard,
    "LunaYAMLWildcardBatch": LunaYAMLWildcardBatch,
    "LunaYAMLWildcardExplorer": LunaYAMLWildcardExplorer,
    "LunaWildcardBuilder": LunaWildcardBuilder,
    "LunaLoRARandomizer": LunaLoRARandomizer,
    "LunaYAMLInjector": LunaYAMLInjector,
    "LunaYAMLPathExplorer": LunaYAMLPathExplorer,
    # Wildcard Connections
    "LunaConnectionMatcher": LunaConnectionMatcher,
    "LunaConnectionEditor": LunaConnectionEditor,
    "LunaSmartLoRALinker": LunaSmartLoRALinker,
    "LunaConnectionStats": LunaConnectionStats,
    # Civitai
    "LunaCivitaiScraper": LunaCivitaiScraper,
    "LunaCivitaiBatchScraper": LunaCivitaiBatchScraper,
    # Expression Pack
    "LunaExpressionPromptBuilder": LunaExpressionPromptBuilder,
    "LunaExpressionSlicerSaver": LunaExpressionSlicerSaver,
    # Daemon proxy loaders
    "LunaDaemonVAELoader": LunaDaemonVAELoader,
    "LunaDaemonCLIPLoader": LunaDaemonCLIPLoader,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    # Config
    "LunaConfigGateway": "Luna Config Gateway",
    # Image saving
    "LunaMultiSaver": "Luna Multi Image Saver",
    # YAML Wildcards
    "LunaYAMLWildcard": "Luna YAML Wildcard",
    "LunaYAMLWildcardBatch": "Luna YAML Wildcard Batch",
    "LunaYAMLWildcardExplorer": "Luna YAML Wildcard Explorer",
    "LunaWildcardBuilder": "Luna Wildcard Builder",
    "LunaLoRARandomizer": "Luna LoRA Randomizer",
    "LunaYAMLInjector": "Luna YAML Injector",
    "LunaYAMLPathExplorer": "Luna YAML Path Explorer",
    # Wildcard Connections
    "LunaConnectionMatcher": "Luna Connection Matcher",
    "LunaConnectionEditor": "Luna Connection Editor",
    "LunaSmartLoRALinker": "Luna Smart LoRA Linker",
    "LunaConnectionStats": "Luna Connection Stats",
    # Civitai
    "LunaCivitaiScraper": "Luna Civitai Metadata Scraper",
    "LunaCivitaiBatchScraper": "Luna Civitai Batch Scraper",
    # Expression Pack
    "LunaExpressionPromptBuilder": "Luna Expression Prompt Builder",
    "LunaExpressionSlicerSaver": "Luna Expression Slicer & Saver",
    # Daemon proxy loaders
    "LunaDaemonVAELoader": "Luna Daemon VAE Loader",
    "LunaDaemonCLIPLoader": "Luna Daemon CLIP Loader",
})
