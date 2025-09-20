import os
import random
import folder_paths
from nodes import MAX_RESOLUTION
import sys

# Add validation to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "validation"))
from validation import luna_validator, validate_node_input

class LunaTextProcessor:
    CATEGORY = "Luna/Text"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("text", "index", "filename")
    FUNCTION = "process_text"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_file": ("STRING", {"default": "", "tooltip": "Path to text file containing lines to process"}),
                "selection_mode": (["random", "sequential"], {"default": "sequential"}),
                "reset_counter": ("BOOLEAN", {"default": False, "label_on": "Reset to Start", "label_off": "Continue"}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "stop_index": ("INT", {"default": -1, "min": -1, "max": 10000, "tooltip": "Stop index (-1 for end of file)"}),
                "step": ("INT", {"default": 1, "min": 1, "max": 100}),
                "additional_text": ("STRING", {"multiline": True, "default": "", "tooltip": "Additional text to concatenate"}),
                "delimiter": ("STRING", {"default": ", ", "tooltip": "Delimiter to use between text file content and additional text"}),
                "concat_mode": (["prepend", "append"], {"default": "append", "tooltip": "Prepend or append additional text"}),
            }
        }

    def __init__(self):
        self.current_index = 0
        self.lines_cache = []
        self.file_path_cache = ""

    def load_text_file(self, file_path):
        """Load and cache text file lines"""
        # Validate file path
        if file_path:
            try:
                validated_path = luna_validator.validate_file_path(
                    file_path,
                    must_exist=True,
                    allowed_extensions=['.txt', '.csv', '.md'],
                    cache_key=f"text_file_{hash(file_path)}"
                )
                file_path = validated_path
            except ValueError as e:
                print(f"[LunaTextProcessor] {e}")
                return []

        if not file_path or not os.path.exists(file_path):
            return []

        # Check if file has changed
        if file_path != self.file_path_cache:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.lines_cache = [line.strip() for line in f.readlines() if line.strip()]
                self.file_path_cache = file_path
            except Exception as e:
                print(f"[LunaTextProcessor] Error loading file {file_path}: {e}")
                return []
        return self.lines_cache

    def get_line_by_index(self, lines, index, start_idx, stop_idx, step):
        """Get line by index with bounds checking"""
        if not lines:
            return "", 0

        # Apply start/stop bounds
        if stop_idx == -1:
            stop_idx = len(lines)

        # Create valid indices list
        valid_indices = list(range(max(0, start_idx), min(len(lines), stop_idx), step))
        if not valid_indices:
            return "", 0

        # Ensure index is within valid range
        if index >= len(valid_indices):
            index = 0

        actual_index = valid_indices[index]
        return lines[actual_index], actual_index

    def get_random_line(self, lines, start_idx, stop_idx):
        """Get random line within bounds"""
        if not lines:
            return "", 0

        # Apply start/stop bounds
        if stop_idx == -1:
            stop_idx = len(lines)

        valid_indices = list(range(max(0, start_idx), min(len(lines), stop_idx)))
        if not valid_indices:
            return "", 0

        actual_index = random.choice(valid_indices)
        return lines[actual_index], actual_index

    def process_text(self, text_file, selection_mode, reset_counter, start_index, stop_index, step, additional_text, delimiter, concat_mode):
        # Validate additional text
        try:
            additional_text = luna_validator.validate_text_input(
                additional_text,
                max_length=10000,  # Reasonable limit
                allow_empty=True,
                cache_key=f"additional_text_{hash(additional_text)}"
            )
        except ValueError as e:
            print(f"[LunaTextProcessor] {e}")
            additional_text = ""

        # Load text file
        lines = self.load_text_file(text_file)

        # Extract filename without extension
        filename = ""
        if text_file:
            filename = os.path.splitext(os.path.basename(text_file))[0]

        # Reset counter if requested
        if reset_counter:
            self.current_index = 0

        # Get line based on selection mode
        if selection_mode == "random":
            selected_line, actual_index = self.get_random_line(lines, start_index, stop_index)
        else:  # sequential
            selected_line, actual_index = self.get_line_by_index(lines, self.current_index, start_index, stop_index, step)
            # Increment counter for next call
            self.current_index += 1

        # Concatenate additional text
        if additional_text:
            if concat_mode == "prepend":
                final_text = additional_text + delimiter + selected_line
            else:  # append
                final_text = selected_line + delimiter + additional_text
        else:
            final_text = selected_line

        # Create display strings for UI
        current_index_display = f"Current Index: {actual_index}"
        concatenated_display = f"Output: {final_text[:100]}{'...' if len(final_text) > 100 else ''}"

        return (final_text, actual_index, filename)

NODE_CLASS_MAPPINGS = {
    "LunaTextProcessor": LunaTextProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaTextProcessor": "Luna Text Processor",
}