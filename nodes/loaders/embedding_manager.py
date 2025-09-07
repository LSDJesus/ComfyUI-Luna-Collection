import folder_paths

# ===================================================================================
# G L O B A L S
# ===================================================================================

MAX_EMBEDDING_SLOTS = 4

# ===================================================================================
# THE LEXICON NODE: LUNA EMBEDDING MANAGER
# ===================================================================================

class LunaEmbeddingManager:
    """
    The Luna Embedding Manager. A tool for building a string of precisely
    weighted embeddings for use in a CLIP Text Encode node. Each embedding can
    be individually toggled on or off for rapid experimentation.
    """
    CATEGORY = "Luna/Loaders"
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("embedding_string",)
    FUNCTION = "format_embeddings"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }
        
        for i in range(1, MAX_EMBEDDING_SLOTS + 1):
            inputs["required"][f"embedding_{i}_enabled"] = ("BOOLEAN", {"default": True})
            # We add "None" to the list to allow for empty slots.
            inputs["required"][f"embedding_name_{i}"] = (["None"] + folder_paths.get_filename_list("embeddings"), )
            inputs["required"][f"embedding_weight_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            
        return inputs

    def format_embeddings(self, enabled, **kwargs):
        """
        The core function. It assembles the final, formatted embedding string.
        """
        if not enabled:
            return ("",)

        embedding_parts = []
        for i in range(1, MAX_EMBEDDING_SLOTS + 1):
            embedding_enabled = kwargs.get(f"embedding_{i}_enabled", False)
            if embedding_enabled:
                embedding_name = kwargs.get(f"embedding_name_{i}")
                embedding_weight = kwargs.get(f"embedding_weight_{i}", 1.0)
                
                if embedding_name and embedding_name != "None":
                    # The format for weighted embeddings is (name:weight).
                    # ComfyUI's Text Encode node handles the "embedding:" prefix automatically.
                    formatted_part = f"({embedding_name}:{embedding_weight})"
                    embedding_parts.append(formatted_part)
        
        # Join the parts with a comma and a space for readability.
        final_string = ", ".join(embedding_parts)

        return (final_string,)

# ===================================================================================
# N O D E   R E G I S T R A T I O N
# ===================================================================================

NODE_CLASS_MAPPINGS = {
    "LunaEmbeddingManager": LunaEmbeddingManager
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaEmbeddingManager": "Luna Embedding Manager"
}