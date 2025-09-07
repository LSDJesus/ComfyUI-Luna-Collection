import random
import math
import folder_paths

# ===================================================================================
# G L O B A L S
# ===================================================================================

# We still define the max slots here to keep the node consistent.
MAX_LORA_SLOTS = 4

# ===================================================================================
# H E L P E R   F U N C T I O N
# ===================================================================================

def get_random_weight(min_w, max_w, precision, seeded_random):
    """
    Generates a random float within a specified range and with a given precision.
    This ensures that our random numbers are clean and follow the user's intent.
    """
    if min_w > max_w:
        min_w, max_w = max_w, min_w # Swap if min is greater than max

    if precision == 0: # Avoid division by zero
        return min_w

    # Calculate the number of possible steps between min and max
    steps = int((max_w - min_w) / precision)
    if steps == 0:
        return min_w

    # Choose a random step
    random_step = seeded_random.randint(0, steps)
    
    # Calculate the final random value
    random_value = min_w + random_step * precision

    # Get the number of decimal places from the precision to ensure clean rounding
    decimal_places = 0
    if '.' in str(precision):
        decimal_places = len(str(precision).split('.')[1])

    return round(random_value, decimal_places)


# ===================================================================================
# THE ORACLE NODE: LUNA LORA STACKER (RANDOM)
# ===================================================================================

class LunaLoRAStackerRandom:
    """
    The Random Luna LoRA Stacker. It configures a stack of LoRAs where each
    LoRA's weight is individually randomized based on user-defined min/max/precision
    settings, all controlled by a single master seed for reproducibility.
    """
    CATEGORY = "Luna/Loaders"
    
    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("LORA_STACK",)
    FUNCTION = "configure_stack"

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the inputs, including the master seed and per-LoRA random ranges.
        """
        inputs = {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "show_previews": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "optional_lora_stack": ("LORA_STACK",)
            }
        }
        
        for i in range(1, MAX_LORA_SLOTS + 1):
            inputs["required"][f"lora_{i}_enabled"] = ("BOOLEAN", {"default": True})
            inputs["required"][f"lora_name_{i}"] = (["None"] + folder_paths.get_filename_list("loras"), )
            inputs["required"][f"min_model_strength_{i}"] = ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01})
            inputs["required"][f"max_model_strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            inputs["required"][f"precision_model_{i}"] = ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01})
            
            # We hide the clip strength options by default for a cleaner UI.
            # A future JS update could show/hide these based on a simple/advanced toggle.
            inputs["hidden"] = {
                "min_clip_strength_{i}": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "max_clip_strength_{i}": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "precision_clip_{i}": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
            }
            
        return inputs

    def configure_stack(self, enabled, show_previews, seed, optional_lora_stack=None, **kwargs):
        """
        The core function. It assembles the LORA_STACK with randomized weights.
        """
        if not enabled:
            return (optional_lora_stack if optional_lora_stack is not None else [],)

        lora_stack = optional_lora_stack if optional_lora_stack is not None else []
        
        # Create a single random number generator instance with our master seed.
        # This is the key to reproducible chaos.
        seeded_random = random.Random()
        seeded_random.seed(seed)

        for i in range(1, MAX_LORA_SLOTS + 1):
            lora_enabled = kwargs.get(f"lora_{i}_enabled", False)
            if lora_enabled:
                lora_name = kwargs.get(f"lora_name_{i}")
                if lora_name and lora_name != "None":
                    
                    # Get the min/max/precision for this specific LoRA
                    min_model = kwargs.get(f"min_model_strength_{i}", 0.0)
                    max_model = kwargs.get(f"max_model_strength_{i}", 1.0)
                    prec_model = kwargs.get(f"precision_model_{i}", 0.1)

                    min_clip = kwargs.get(f"min_clip_strength_{i}", 0.0)
                    max_clip = kwargs.get(f"max_clip_strength_{i}", 1.0)
                    prec_clip = kwargs.get(f"precision_clip_{i}", 0.1)

                    # Generate the random weights using our seeded generator
                    random_model_strength = get_random_weight(min_model, max_model, prec_model, seeded_random)
                    random_clip_strength = get_random_weight(min_clip, max_clip, prec_clip, seeded_random)
                    
                    lora_stack.append((lora_name, random_model_strength, random_clip_strength))

        return (lora_stack,)

# ===================================================================================
# N O D E   R E G I S T R A T I O N
# ===================================================================================
# We do not need to redefine the web endpoint. The JavaScript for this node
# will call the exact same '/luna/get_lora_metadata' endpoint as the Advanced Stacker.
# This is the beauty of our modular design.

NODE_CLASS_MAPPINGS = {
    "LunaLoRAStackerRandom": LunaLoRAStackerRandom
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaLoRAStackerRandom": "Luna LoRA Stacker (Random)"
}