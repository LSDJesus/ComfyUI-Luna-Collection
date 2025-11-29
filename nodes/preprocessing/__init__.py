from .luna_prompt_preprocessor import LunaPromptPreprocessor
from .luna_logic_resolver import LunaLogicResolver

NODE_CLASS_MAPPINGS = {
    "LunaPromptPreprocessor": LunaPromptPreprocessor,
    "LunaLogicResolver": LunaLogicResolver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaPromptPreprocessor": "Luna Prompt Preprocessor",
    "LunaLogicResolver": "Luna Logic Resolver",
}