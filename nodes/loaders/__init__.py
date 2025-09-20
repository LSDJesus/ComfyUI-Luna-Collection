from .luna_checkpoint_loader import LunaCheckpointLoader
from .luna_embedding_manager import LunaEmbeddingManager
from .luna_embedding_manager_random import LunaEmbeddingManagerRandom
from .luna_lora_stacker import LunaLoRAStacker
from .luna_lora_stacker_random import LunaLoRAStackerRandom

NODE_CLASS_MAPPINGS = {
    "LunaCheckpointLoader": LunaCheckpointLoader,
    "LunaEmbeddingManager": LunaEmbeddingManager,
    "LunaEmbeddingManagerRandom": LunaEmbeddingManagerRandom,
    "LunaLoRAStacker": LunaLoRAStacker,
    "LunaLoRAStackerRandom": LunaLoRAStackerRandom,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaCheckpointLoader": "Luna Checkpoint Loader",
    "LunaEmbeddingManager": "Luna Embedding Manager",
    "LunaEmbeddingManagerRandom": "Luna Embedding Manager Random",
    "LunaLoRAStacker": "Luna LoRA Stacker",
    "LunaLoRAStackerRandom": "Luna LoRA Stacker Random",
}
