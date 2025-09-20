# Import from subdirectories
from . import preprocessing
from . import performance
from . import upscaling
from . import detailing
from . import loaders

# Import individual files
from .luna_image_caption import luna_image_caption
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
from .luna_yolo_annotation_exporter import Luna_YOLO_Annotation_Exporter

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

# Add mappings from individual files
NODE_CLASS_MAPPINGS.update({
    "luna_image_caption": luna_image_caption,
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
    "Luna_YOLO_Annotation_Exporter": Luna_YOLO_Annotation_Exporter,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "luna_image_caption": "Luna Image Caption",
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
    "Luna_YOLO_Annotation_Exporter": "Luna YOLO Annotation Exporter",
})
