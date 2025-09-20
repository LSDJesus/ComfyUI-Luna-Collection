from .luna_performance_concat import LunaPerformanceStatsConcat
from .luna_performance_condition import LunaPerformanceCondition
from .luna_performance_display import LunaPerformanceDisplay
from .luna_performance_logger import LunaPerformanceLogger

NODE_CLASS_MAPPINGS = {
    "LunaPerformanceStatsConcat": LunaPerformanceStatsConcat,
    "LunaPerformanceCondition": LunaPerformanceCondition,
    "LunaPerformanceDisplay": LunaPerformanceDisplay,
    "LunaPerformanceLogger": LunaPerformanceLogger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaPerformanceStatsConcat": "Luna Performance Stats Concat",
    "LunaPerformanceCondition": "Luna Performance Condition",
    "LunaPerformanceDisplay": "Luna Performance Display",
    "LunaPerformanceLogger": "Luna Performance Logger",
}