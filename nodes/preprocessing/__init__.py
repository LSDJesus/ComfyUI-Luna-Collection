from .luna_prompt_preprocessor import LunaPromptPreprocessor
from .luna_text_processor import LunaTextProcessor
from .luna_unified_prompt_processor import LunaUnifiedPromptProcessor

NODE_CLASS_MAPPINGS = {
    "LunaPromptPreprocessor": LunaPromptPreprocessor,
    "LunaTextProcessor": LunaTextProcessor,
    "LunaUnifiedPromptProcessor": LunaUnifiedPromptProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaPromptPreprocessor": "Luna Prompt Preprocessor",
    "LunaTextProcessor": "Luna Text Processor",
    "LunaUnifiedPromptProcessor": "Luna Unified Prompt Processor",
}