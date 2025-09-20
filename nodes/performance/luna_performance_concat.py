from typing import Dict, Any, List
import time

class LunaPerformanceStatsConcat:
    """
    Concatenates multiple performance_stats inputs into a combined statistics report
    """
    CATEGORY = "Luna/Utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "performance_stats_1": ("PERFORMANCE_STATS", {"tooltip": "First performance stats input"}),
                "performance_stats_2": ("PERFORMANCE_STATS", {"tooltip": "Second performance stats input"}),
                "performance_stats_3": ("PERFORMANCE_STATS", {"tooltip": "Third performance stats input"}),
                "performance_stats_4": ("PERFORMANCE_STATS", {"tooltip": "Fourth performance stats input"}),
                "performance_stats_5": ("PERFORMANCE_STATS", {"tooltip": "Fifth performance stats input"}),
                "performance_stats_6": ("PERFORMANCE_STATS", {"tooltip": "Sixth performance stats input"}),
                "performance_stats_7": ("PERFORMANCE_STATS", {"tooltip": "Seventh performance stats input"}),
                "performance_stats_8": ("PERFORMANCE_STATS", {"tooltip": "Eighth performance stats input"}),
                "performance_stats_9": ("PERFORMANCE_STATS", {"tooltip": "Ninth performance stats input"}),
                "performance_stats_10": ("PERFORMANCE_STATS", {"tooltip": "Tenth performance stats input"}),
            }
        }

    RETURN_TYPES = ("PERFORMANCE_STATS", "STRING")
    RETURN_NAMES = ("combined_stats", "summary_report")
    FUNCTION = "concatenate_stats"

    def concatenate_stats(self, **kwargs):
        """Concatenate multiple performance stats into a combined report"""
        # Filter out None values and collect all performance stats
        performance_stats_list = []
        node_names = []

        for key, value in kwargs.items():
            if key.startswith("performance_stats_") and value is not None and isinstance(value, dict):
                performance_stats_list.append(value)
                node_names.append(value.get("node_name", f"Node_{key.split('_')[-1]}"))

        if not performance_stats_list:
            empty_stats = {
                "error": "No performance stats provided",
                "timestamp": time.time(),
                "node_count": 0
            }
            return (empty_stats, "No performance data available")

        # Combine all stats
        combined_stats = self._combine_performance_stats(performance_stats_list, node_names)

        # Generate summary report
        summary_report = self._generate_summary_report(combined_stats, node_names)

        return (combined_stats, summary_report)

    def _combine_performance_stats(self, stats_list: List[Dict[str, Any]], node_names: List[str]) -> Dict[str, Any]:
        """Combine multiple performance stats into aggregated metrics"""
        if not stats_list:
            return {}

        combined = {
            "combined_timestamp": time.time(),
            "total_nodes": len(stats_list),
            "node_names": node_names,
            "individual_stats": stats_list,
            "aggregated_metrics": {}
        }

        # Aggregate numeric metrics
        numeric_keys = [
            "processing_time", "vram_usage_mb", "vram_delta_mb",
            "system_memory_percent", "cpu_percent",
            "gpu_memory_allocated_mb", "gpu_memory_reserved_mb"
        ]

        for key in numeric_keys:
            values = [stats.get(key, 0) for stats in stats_list if isinstance(stats.get(key), (int, float))]
            if values:
                combined["aggregated_metrics"][f"{key}_total"] = sum(values)
                combined["aggregated_metrics"][f"{key}_average"] = sum(values) / len(values)
                combined["aggregated_metrics"][f"{key}_min"] = min(values)
                combined["aggregated_metrics"][f"{key}_max"] = max(values)

        # Calculate total processing time
        total_time = sum(stats.get("processing_time", 0) for stats in stats_list)
        combined["aggregated_metrics"]["total_pipeline_time"] = total_time

        # Calculate efficiency metrics
        if len(stats_list) > 1:
            avg_time_per_node = total_time / len(stats_list)
            combined["aggregated_metrics"]["average_time_per_node"] = avg_time_per_node

            # Calculate parallelization efficiency (if timestamps are available)
            timestamps = [stats.get("timestamp", 0) for stats in stats_list]
            if all(t > 0 for t in timestamps):
                time_span = max(timestamps) - min(timestamps)
                if time_span > 0:
                    combined["aggregated_metrics"]["parallelization_efficiency"] = total_time / time_span

        return combined

    def _generate_summary_report(self, combined_stats: Dict[str, Any], node_names: List[str]) -> str:
        """Generate a human-readable summary report"""
        if not combined_stats or "error" in combined_stats:
            return "No performance data available"

        lines = []
        lines.append("╔══════════════════════════════════════════════════════════════╗")
        lines.append("║                 LUNA PIPELINE PERFORMANCE REPORT              ║")
        lines.append("╠══════════════════════════════════════════════════════════════╣")

        # Basic info
        total_nodes = combined_stats.get("total_nodes", 0)
        lines.append(f"║ Total Nodes Processed: {total_nodes:<35} ║")

        # Time metrics
        total_time = combined_stats.get("aggregated_metrics", {}).get("total_pipeline_time", 0)
        avg_time = combined_stats.get("aggregated_metrics", {}).get("average_time_per_node", 0)
        lines.append(f"║ Total Pipeline Time: {total_time:.2f}s{'':<28} ║")
        lines.append(f"║ Average Time/Node: {avg_time:.2f}s{'':<30} ║")

        # Memory metrics
        total_vram = combined_stats.get("aggregated_metrics", {}).get("vram_usage_mb_total", 0)
        max_vram = combined_stats.get("aggregated_metrics", {}).get("vram_usage_mb_max", 0)
        lines.append(f"║ Peak VRAM Usage: {max_vram:.1f}MB{'':<32} ║")
        lines.append(f"║ Total VRAM Delta: {total_vram:.1f}MB{'':<31} ║")

        # System metrics
        avg_cpu = combined_stats.get("aggregated_metrics", {}).get("cpu_percent_average", 0)
        avg_mem = combined_stats.get("aggregated_metrics", {}).get("system_memory_percent_average", 0)
        lines.append(f"║ Average CPU Usage: {avg_cpu:.1f}%{'':<31} ║")
        lines.append(f"║ Average Memory: {avg_mem:.1f}%{'':<33} ║")

        # Node list
        lines.append("╠══════════════════════════════════════════════════════════════╣")
        lines.append("║ Node Breakdown:                                              ║")
        for i, name in enumerate(node_names[:5]):  # Show first 5 nodes
            lines.append(f"║  {i+1}. {name:<52} ║")
        if len(node_names) > 5:
            lines.append(f"║  ... and {len(node_names)-5} more nodes{'':<33} ║")

        lines.append("╚══════════════════════════════════════════════════════════════╝")

        return "\n".join(lines)