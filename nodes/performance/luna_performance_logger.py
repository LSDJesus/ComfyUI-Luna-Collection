import os
import json
from datetime import datetime
import folder_paths

# Import Luna validation system
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from validation import luna_validator, validate_node_input
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    validate_node_input = None

def conditional_validate(*args, **kwargs):
    """Conditionally apply validation decorator."""
    def decorator(func):
        if VALIDATION_AVAILABLE and validate_node_input:
            return validate_node_input(*args, **kwargs)(func)
        return func
    return decorator

class LunaPerformanceLogger:
    """
    Logs performance statistics from LunaSampler to JSON files
    """
    CATEGORY = "Luna/Utils"
    RETURN_TYPES = ()
    FUNCTION = "log_performance"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "performance_stats": ("PERFORMANCE_STATS", {"tooltip": "Performance statistics from LunaSampler"}),
                "log_filename": ("STRING", {"default": "performance_log.json", "tooltip": "Name of the log file"}),
                "append_mode": ("BOOLEAN", {"default": True, "label_on": "Append", "label_off": "Overwrite", "tooltip": "Append to existing log or overwrite"})
            },
            "optional": {
                "custom_log_dir": ("STRING", {"default": "", "tooltip": "Custom log directory (leave empty for default)"})
            }
        }

    @conditional_validate('log_filename', max_length=255)
    def log_performance(self, performance_stats, log_filename="performance_log.json", append_mode=True, custom_log_dir=""):
        # Handle combined stats from LunaPerformanceStatsConcat
        if isinstance(performance_stats, dict) and "combined_timestamp" in performance_stats:
            # This is combined stats - log each individual stat separately
            individual_stats = performance_stats.get("individual_stats", [])
            for i, stats in enumerate(individual_stats):
                self._log_single_stats(stats, f"{log_filename.replace('.json', f'_node_{i+1}.json')}", append_mode, custom_log_dir)

            # Also log the combined summary
            self._log_single_stats(performance_stats, f"{log_filename.replace('.json', '_combined.json')}", append_mode, custom_log_dir)
            print(f"[LunaPerformanceLogger] Logged {len(individual_stats)} individual stats + combined summary")
            return ()

        # Handle single stats
        self._log_single_stats(performance_stats, log_filename, append_mode, custom_log_dir)
        return ()

    def _log_single_stats(self, performance_stats, log_filename, append_mode, custom_log_dir):
        """Log a single performance stats entry"""
        # Determine log directory
        if custom_log_dir:
            log_dir = os.path.join(folder_paths.get_output_directory(), custom_log_dir)
        else:
            log_dir = os.path.join(folder_paths.get_output_directory(), "performance_logs")

        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_filename)

        # Prepare log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "performance_data": performance_stats
        }

        # Load existing log if appending
        if append_mode and os.path.exists(log_path):
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = [existing_data]
            except:
                existing_data = []
        else:
            existing_data = []

        # Add new entry
        existing_data.append(log_entry)

        # Save log
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)

        print(f"[LunaPerformanceLogger] Performance stats logged to: {log_path}")