import json

class LunaPerformanceDisplay:
    """
    Displays performance statistics from LunaSampler in a readable format
    """
    CATEGORY = "Luna/Utils"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("formatted_stats",)
    FUNCTION = "display_performance"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "performance_stats": ("PERFORMANCE_STATS", {"tooltip": "Performance statistics from LunaSampler"}),
                "display_format": (["compact", "detailed", "json"], {"default": "compact", "tooltip": "Format for displaying performance stats"})
            }
        }

    def display_performance(self, performance_stats, display_format="compact"):
        # Handle combined stats from LunaPerformanceStatsConcat
        if isinstance(performance_stats, dict) and "combined_timestamp" in performance_stats:
            return self._display_combined_stats(performance_stats, display_format)

        # Handle single stats
        return self._display_single_stats(performance_stats, display_format)

    def _display_single_stats(self, performance_stats, display_format):
        """Display single performance stats"""
        if display_format == "json":
            return (json.dumps(performance_stats, indent=2, ensure_ascii=False),)

        # Extract key metrics
        processing_time = performance_stats.get('processing_time', 0)
        vram_usage = performance_stats.get('vram_usage_mb', 0)
        vram_delta = performance_stats.get('vram_delta_mb', 0)
        node_name = performance_stats.get('node_name', 'Unknown')
        system_memory = performance_stats.get('system_memory_percent', 0)
        cpu_percent = performance_stats.get('cpu_percent', 0)

        if display_format == "compact":
            formatted = f"[{node_name}] Time: {processing_time:.2f}s | VRAM: {vram_usage:.1f}MB (Δ{vram_delta:+.1f}MB) | CPU: {cpu_percent:.1f}% | Mem: {system_memory:.1f}%"
        else:  # detailed
            formatted = f"""
╔══════════════════════════════════════════════════════════════╗
║                    {node_name.upper():<52} ║
╠══════════════════════════════════════════════════════════════╣
║ Processing Time: {processing_time:.2f}s{'':<41} ║
║ VRAM Usage: {vram_usage:.1f}MB (Δ{vram_delta:+.1f}MB){'':<35} ║
║ CPU Usage: {cpu_percent:.1f}%{'':<47} ║
║ System Memory: {system_memory:.1f}%{'':<42} ║
╚══════════════════════════════════════════════════════════════╝
"""

        return (formatted.strip(),)

    def _display_combined_stats(self, performance_stats, display_format):
        """Display combined performance stats"""
        if display_format == "json":
            return (json.dumps(performance_stats, indent=2, ensure_ascii=False),)

        total_nodes = performance_stats.get("total_nodes", 0)
        total_time = performance_stats.get("aggregated_metrics", {}).get("total_pipeline_time", 0)
        avg_time = performance_stats.get("aggregated_metrics", {}).get("average_time_per_node", 0)
        max_vram = performance_stats.get("aggregated_metrics", {}).get("vram_usage_mb_max", 0)
        avg_cpu = performance_stats.get("aggregated_metrics", {}).get("cpu_percent_average", 0)

        if display_format == "compact":
            formatted = f"Combined Stats: {total_nodes} nodes | Total: {total_time:.2f}s | Avg: {avg_time:.2f}s | Peak VRAM: {max_vram:.1f}MB | Avg CPU: {avg_cpu:.1f}%"
        else:  # detailed
            node_names = performance_stats.get("node_names", [])
            formatted = f"""
╔══════════════════════════════════════════════════════════════╗
║                 COMBINED PIPELINE PERFORMANCE                  ║
╠══════════════════════════════════════════════════════════════╣
║ Total Nodes: {total_nodes:<50} ║
║ Total Processing Time: {total_time:.2f}s{'':<35} ║
║ Average Time per Node: {avg_time:.2f}s{'':<35} ║
║ Peak VRAM Usage: {max_vram:.1f}MB{'':<42} ║
║ Average CPU Usage: {avg_cpu:.1f}%{'':<41} ║
╠══════════════════════════════════════════════════════════════╣
║ Node Summary:{'':<52} ║"""

            # Add node list
            for i, name in enumerate(node_names[:3]):  # Show first 3 nodes
                formatted += f"\n║  {i+1}. {name:<54} ║"
            if len(node_names) > 3:
                formatted += f"\n║  ... and {len(node_names)-3} more{'':<46} ║"

            formatted += "\n╚══════════════════════════════════════════════════════════════╝"

        return (formatted.strip(),)

NODE_CLASS_MAPPINGS = {
    "LunaPerformanceDisplay": LunaPerformanceDisplay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaPerformanceDisplay": "Luna Performance Display",
}