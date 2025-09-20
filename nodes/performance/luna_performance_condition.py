class LunaPerformanceCondition:
    """
    Makes conditional decisions based on performance statistics
    """
    CATEGORY = "Luna/Utils"
    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("condition_met", "reason")
    FUNCTION = "check_condition"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "performance_stats": ("PERFORMANCE_STATS", {"tooltip": "Performance statistics from LunaSampler"}),
                "condition_type": (["time_threshold", "vram_threshold", "memory_threshold", "error_check"], {"default": "time_threshold"}),
                "threshold_value": ("FLOAT", {"default": 10.0, "min": 0.0, "step": 0.1, "tooltip": "Threshold value for the condition"}),
                "comparison": (["greater_than", "less_than", "equal_to"], {"default": "less_than", "tooltip": "Comparison operator"})
            }
        }

    def check_condition(self, performance_stats, condition_type, threshold_value, comparison):
        # Extract the relevant metric
        if condition_type == "time_threshold":
            value = performance_stats.get('sampling_time', 0)
            metric_name = "sampling time"
            unit = "seconds"
        elif condition_type == "vram_threshold":
            value = performance_stats.get('vram_usage_mb', 0)
            metric_name = "VRAM usage"
            unit = "MB"
        elif condition_type == "memory_threshold":
            value = performance_stats.get('system_memory_percent', 0)
            metric_name = "system memory"
            unit = "%"
        elif condition_type == "error_check":
            has_error = 'error' in performance_stats and performance_stats['error']
            condition_met = not has_error  # True if no error
            reason = f"No errors detected" if condition_met else f"Error found: {performance_stats.get('error', 'Unknown error')}"
            return (condition_met, reason)
        else:
            return (False, f"Unknown condition type: {condition_type}")

        # Perform comparison
        if comparison == "greater_than":
            condition_met = value > threshold_value
            comparison_text = f">{threshold_value}"
        elif comparison == "less_than":
            condition_met = value < threshold_value
            comparison_text = f"<{threshold_value}"
        else:  # equal_to
            condition_met = abs(value - threshold_value) < 0.01
            comparison_text = f"≈{threshold_value}"

        reason = f"{metric_name}: {value:.2f}{unit} {comparison_text} {'✓' if condition_met else '✗'}"

        return (condition_met, reason)