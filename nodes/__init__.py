# Luna Collection - Node Registration
# Auto-discovers all node files and registers them with ComfyUI

# Import from subdirectories
from . import upscaling
from .prompting import promptcraft

# Workflow nodes
from .workflow.luna_config_gateway import LunaConfigGateway
from .workflow.luna_multi_saver import LunaMultiSaver
from .workflow.luna_expression_pack import (
    LunaExpressionPromptBuilder,
    LunaExpressionSlicerSaver,
)

# Prompting nodes
from .prompting.luna_yaml_wildcard import (
    LunaYAMLWildcard,
    LunaYAMLWildcardBatch,
    LunaYAMLWildcardExplorer,
    LunaWildcardBuilder,
    LunaLoRARandomizer,
    LunaYAMLInjector,
    LunaYAMLPathExplorer,
)
from .prompting.luna_wildcard_connections import (
    LunaConnectionMatcher,
    LunaConnectionEditor,
    LunaSmartLoRALinker,
    LunaConnectionStats,
)
from .prompting.luna_batch_prompt_extractor import (
    LunaBatchPromptExtractor,
    LunaBatchPromptLoader,
    LunaDimensionScaler,
)
from .prompting.luna_trigger_injector import LunaLoRATriggerInjector

# Loader nodes
from .loaders.luna_daemon_loader import (
    LunaDaemonVAELoader,
    LunaDaemonCLIPLoader,
)
from .loaders.luna_model_router import LunaModelRouter
from .loaders.luna_dynamic_loader import LunaDynamicModelLoader, LunaOptimizedWeightsManager
from .loaders.luna_secondary_loader import LunaSecondaryModelLoader, LunaModelRestore

# Vision nodes
from .vision.luna_vision_node import LunaVisionNode
from .vision.luna_vlm_prompt_generator import LunaVLMPromptGenerator
from .vision.luna_zimage_encoder import LunaZImageEncoder
from .vision.luna_zimage_processor import LunaZImageProcessor

# Utility nodes
from .utilities.luna_civitai_scraper import (
    LunaCivitaiScraper,
    LunaCivitaiBatchScraper,
)
from .utilities.luna_lora_validator import LunaLoRAValidator

# Daemon API - registers web routes for the daemon panel
try:
    from .utilities import luna_daemon_api  # noqa: F401 - imported for side effects (route registration)
except ImportError:
    pass

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
    # Batch prompt utilities
    "LunaBatchPromptExtractor": LunaBatchPromptExtractor,
    "LunaBatchPromptLoader": LunaBatchPromptLoader,
    "LunaDimensionScaler": LunaDimensionScaler,
    # LoRA validation
    "LunaLoRAValidator": LunaLoRAValidator,
    # LoRA trigger injection
    "LunaLoRATriggerInjector": LunaLoRATriggerInjector,
    # Model loading
    "LunaModelRouter": LunaModelRouter,
    "LunaDynamicModelLoader": LunaDynamicModelLoader,
    "LunaOptimizedWeightsManager": LunaOptimizedWeightsManager,
    # Vision and VLM
    "LunaVisionNode": LunaVisionNode,
    "LunaVLMPromptGenerator": LunaVLMPromptGenerator,
    # Multi-model workflows
    "LunaSecondaryModelLoader": LunaSecondaryModelLoader,
    "LunaModelRestore": LunaModelRestore,
    # Z-IMAGE
    "LunaZImageEncoder": LunaZImageEncoder,
    "LunaZImageProcessor": LunaZImageProcessor,
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
    # Batch prompt utilities
    "LunaBatchPromptExtractor": "Luna Batch Prompt Extractor",
    "LunaBatchPromptLoader": "Luna Batch Prompt Loader",
    "LunaDimensionScaler": "Luna Dimension Scaler",
    # LoRA validation
    "LunaLoRAValidator": "Luna LoRA Validator",
    # LoRA trigger injection
    "LunaLoRATriggerInjector": "Luna LoRA Trigger Injector",
    # Model loading
    "LunaModelRouter": "Luna Model Router ⚡",
    "LunaDynamicModelLoader": "Luna Dynamic Model Loader",
    "LunaOptimizedWeightsManager": "Luna Optimized Weights Manager",
    # Vision and VLM
    "LunaVisionNode": "Luna Vision Encoder 👁️",
    "LunaVLMPromptGenerator": "Luna VLM Prompt Generator 🤖",
    # Multi-model workflows
    "LunaSecondaryModelLoader": "Luna Secondary Model Loader 🔄",
    "LunaModelRestore": "Luna Model Restore 📤",
    # Z-IMAGE
    "LunaZImageEncoder": "Luna Z-IMAGE Encoder 🌙",
    "LunaZImageProcessor": "Luna Z-IMAGE Processor 🔄",
})

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
