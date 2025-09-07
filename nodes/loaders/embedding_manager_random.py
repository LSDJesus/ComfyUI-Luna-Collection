import random
import folder_paths

# ===================================================================================
# G L O B A L S & H E L P E R S
# ===================================================================================

MAX_EMBEDDING_SLOTS = 4

# We include the same helper function for consistency and self-containment.
def get_random_weight(min_w, max_w, precision, seeded_random):
    if min_w > max_w:
        min_w, max_w = max_w, min_w
    if precision == 0:
        return min_w
    steps = int((max_w - min_w) / precision)
    if steps == 0:
        return min_w
    random_step = seeded_random.randint(0, steps)
    random_value = min_w + random_step * precision
    decimal_places = 0
    if '.' in str(precision):
        decimal_places = len(str(precision).split('.')[1])
    return round(random_value, decimal_places)

# ===================================================================================
# THE RANDOM LEXICON NODE: LUNA EMBEDDING MANAGER (RANDOM)
# ===================================================================================

class LunaEmbeddingManagerRandom:
    """
    The Random Luna Embedding Manager. It builds a string of embeddings where
    each weight is individually randomized based on user-defined min/max/precision
    settings, all controlled by a single master seed.
    """
    CATEGORY = "Luna/Loaders"
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("embedding_string",)
    FUNCTION = "format_random_embeddings"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
        
        for i in range(1, MAX_EMBEDDING_SLOTS + 1):
            inputs["required"][f"embedding_{i}_enabled"] = ("BOOLEAN", {"default": True})
            inputs["required"][f"embedding_name_{i}"] = (["None"] + folder_paths.get_filename_list("embeddings"), )
            inputs["required"][f"min_weight_{i}"] = ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01})
            inputs["required"][f"max_weight_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            inputs["required"][f"precision_{i}"] = ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01})
            
        return inputs

    def format_random_embeddings(self, enabled, seed, **kwargs):
        """
        The core function. It assembles the final string with randomized weights.
        """
        if not enabled:
            return ("",)

        seeded_random = random.Random()
        seeded_random.seed(seed)
        
        embedding_parts = []
        for i in range(1, MAX_EMBEDDING_SLOTS + 1):
            embedding_enabled = kwargs.get(f"embedding_{i}_enabled", False)
            if embedding_enabled:
                embedding_name = kwargs.get(f"embedding_name_{i}")
                if embedding_name and embedding_name != "None":
                    min_w = kwargs.get(f"min_weight_{i}", 0.0)
                    max_w = kwargs.get(f"max_weight_{i}", 1.0)
                    prec = kwargs.get(f"precision_{i}", 0.1)
                    
                    random_weight = get_random_weight(min_w, max_w, prec, seeded_random)
                    
                    formatted_part = f"({embedding_name}:{random_weight})"
                    embedding_parts.append(formatted_part)
        
        final_string = ", ".join(embedding_parts)

        return (final_string,)

# ===================================================================================
# N O D E   R E G I S T R A T I O N
# ===================================================================================

NODE_CLASS_MAPPINGS = {
    "LunaEmbeddingManagerRandom": LunaEmbeddingManagerRandom
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaEmbeddingManagerRandom": "Luna Embedding Manager (Random)"
}