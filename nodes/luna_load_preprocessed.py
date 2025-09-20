import os
import json
import torch
from safetensors.torch import load_file, save_file
import folder_paths
from datetime import datetime
import gzip
import bz2
import lzma
from io import BytesIO

class LunaSelectPromptFolder:
    CATEGORY = "Luna/Preprocessing"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("folder_path", "json_path")
    FUNCTION = "select_prompt_folder"

    @classmethod
    def INPUT_TYPES(cls):
        # Get available prompt folders from models/luna_prompts
        luna_prompts_dir = os.path.join(folder_paths.models_dir, "luna_prompts")
        if os.path.exists(luna_prompts_dir):
            folders = [f for f in os.listdir(luna_prompts_dir) if os.path.isdir(os.path.join(luna_prompts_dir, f))]
            folders.sort()
        else:
            folders = []

        return {
            "required": {
                "prompt_folder": (folders, {"tooltip": "Select a preprocessed prompt folder from models/luna_prompts"}),
            }
        }

    def select_prompt_folder(self, prompt_folder):
        luna_prompts_dir = os.path.join(folder_paths.models_dir, "luna_prompts")
        folder_path = os.path.join(luna_prompts_dir, prompt_folder)

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Prompt folder not found: {folder_path}")

        # Find JSON file in the folder
        json_files = [f for f in os.listdir(folder_path) if f.endswith('_mappings.json')]
        if not json_files:
            raise FileNotFoundError(f"No mappings JSON file found in {folder_path}")

        json_path = os.path.join(folder_path, json_files[0])  # Use first JSON file found

        print(f"[LunaSelectPromptFolder] Selected folder: {folder_path}")
        print(f"[LunaSelectPromptFolder] Using mappings: {json_path}")

        return (folder_path, json_path)

class LunaLoadPreprocessedPrompt:
    CATEGORY = "Luna/Preprocessing"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING", "INT")
    RETURN_NAMES = ("positive_conditioning", "negative_conditioning", "original_prompt", "index")
    FUNCTION = "load_preprocessed_prompt"

    @classmethod
    def INPUT_TYPES(cls):
        # Auto-detect folders created by LunaPromptPreprocessor
        output_dir = folder_paths.get_output_directory()
        available_folders = []

        if os.path.exists(output_dir):
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isdir(item_path):
                    # Check if this looks like a LunaPromptPreprocessor output folder
                    json_files = [f for f in os.listdir(item_path) if f.endswith('_mappings.json')]
                    if json_files:
                        available_folders.append(item)

        available_folders.sort()

        # Get available negative prompt files from models/luna_prompts
        luna_prompts_dir = os.path.join(folder_paths.models_dir, "luna_prompts")
        negative_files = []
        if os.path.exists(luna_prompts_dir):
            for root, dirs, files in os.walk(luna_prompts_dir):
                for file in files:
                    if file.endswith('.safetensors') and not file.startswith('prompt_'):
                        # Get relative path from luna_prompts directory
                        rel_path = os.path.relpath(os.path.join(root, file), luna_prompts_dir)
                        negative_files.append(rel_path)
            negative_files.sort()

        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "tooltip": "Path to the prompt folder containing JSON mappings and safetensors files"}),
                "prompt_key": ("STRING", {"default": "", "tooltip": "Key/name of the prompt to load (from the mappings JSON)"}),
                "negative_prompt_file": (negative_files, {"tooltip": "Select a negative prompt safetensors file from models/luna_prompts"}),
            },
            "optional": {
                "auto_select_folder": (available_folders, {"tooltip": "Auto-detected folders from LunaPromptPreprocessor output"}),
            }
        }

    def load_preprocessed_prompt(self, folder_path, prompt_key, negative_prompt_file, auto_select_folder=None):
        # Use auto-selected folder if provided and folder_path is empty
        if not folder_path and auto_select_folder:
            output_dir = folder_paths.get_output_directory()
            folder_path = os.path.join(output_dir, auto_select_folder)
            print(f"[LunaLoadPreprocessedPrompt] Using auto-selected folder: {folder_path}")

        # Validate inputs
        if not folder_path:
            raise ValueError("Either folder_path must be provided or auto_select_folder must be chosen")

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Find JSON file in the folder
        json_files = [f for f in os.listdir(folder_path) if f.endswith('_mappings.json')]
        if not json_files:
            raise FileNotFoundError(f"No mappings JSON file found in {folder_path}")

        json_path = os.path.join(folder_path, json_files[0])

        # Load mappings
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                mappings = json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading mappings file: {e}")

        # Find the prompt file
        if prompt_key not in mappings:
            available_keys = list(mappings.keys())[:10]  # Show first 10 for debugging
            raise ValueError(f"Prompt key '{prompt_key}' not found in mappings. Available keys: {available_keys}")

        prompt_filepath = mappings[prompt_key]

        # Validate the safetensors file exists
        if not os.path.exists(prompt_filepath):
            raise FileNotFoundError(f"Preprocessed prompt file not found: {prompt_filepath}")

        # Load the positive prompt safetensors file
        try:
            tensors = self._load_compressed_file(prompt_filepath)
        except Exception as e:
            raise ValueError(f"Error loading positive prompt safetensors file: {e}")

        # Extract the positive conditioning and metadata
        if "clip_embeddings" not in tensors:
            raise ValueError("Invalid positive prompt safetensors file: missing 'clip_embeddings' tensor")

        positive_conditioning = tensors["clip_embeddings"]
        original_prompt = tensors.get("original_prompt", prompt_key)
        index = tensors.get("index", -1)

        # Load negative prompt if specified
        negative_conditioning = None
        if negative_prompt_file:
            luna_prompts_dir = os.path.join(folder_paths.models_dir, "luna_prompts")
            negative_filepath = os.path.join(luna_prompts_dir, negative_prompt_file)

            if not os.path.exists(negative_filepath):
                raise FileNotFoundError(f"Negative prompt file not found: {negative_filepath}")

            try:
                neg_tensors = load_file(negative_filepath)
                if "clip_embeddings" in neg_tensors:
                    negative_conditioning = neg_tensors["clip_embeddings"]
                    print(f"[LunaLoadPreprocessedPrompt] Loaded negative prompt from {negative_filepath}")
                else:
                    print(f"[LunaLoadPreprocessedPrompt] Warning: No 'clip_embeddings' found in negative prompt file")
            except Exception as e:
                print(f"[LunaLoadPreprocessedPrompt] Error loading negative prompt: {e}")

        print(f"[LunaLoadPreprocessedPrompt] Loaded positive prompt '{prompt_key}' from {prompt_filepath}")
        print(f"[LunaLoadPreprocessedPrompt] Original prompt: {original_prompt}")
        print(f"[LunaLoadPreprocessedPrompt] Index: {index}")

        return (positive_conditioning, negative_conditioning, original_prompt, index)

    def _load_compressed_file(self, filepath):
        """Load a compressed safetensors file"""
        try:
            # Determine decompression method based on file extension
            if filepath.endswith('.gz'):
                with gzip.open(filepath, 'rb') as f:
                    data = f.read()
            elif filepath.endswith('.bz2'):
                with bz2.open(filepath, 'rb') as f:
                    data = f.read()
            elif filepath.endswith('.xz'):
                with lzma.open(filepath, 'rb') as f:
                    data = f.read()
            else:
                # Not compressed, load normally
                return load_file(filepath)

            # Ensure data is bytes
            if isinstance(data, str):
                data = data.encode('utf-8')

            # Load from decompressed data using safetensors
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(data)
                temp_file.flush()
                tensors = load_file(temp_file.name)

            # Clean up temp file
            try:
                os.unlink(temp_file.name)
            except:
                pass

            return tensors

        except Exception as e:
            print(f"[LunaLoadPreprocessedPrompt] Error loading compressed file {filepath}: {e}")
            # Fallback to normal loading
            return load_file(filepath)

class LunaModifyPreprocessedPrompt:
    CATEGORY = "Luna/Preprocessing"
    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("modified_conditioning", "modified_prompt")
    FUNCTION = "modify_preprocessed_prompt"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "CLIP model for re-encoding modified prompts"}),
                "preprocessed_conditioning": ("CONDITIONING", {"tooltip": "Preprocessed conditioning tensor to modify"}),
                "original_prompt": ("STRING", {"tooltip": "Original prompt text from the preprocessed file"}),
                "prepend_text": ("STRING", {"multiline": True, "default": "", "tooltip": "Text to prepend to the original prompt"}),
                "append_text": ("STRING", {"multiline": True, "default": "", "tooltip": "Text to append to the original prompt"}),
            }
        }

    def modify_preprocessed_prompt(self, clip, preprocessed_conditioning, original_prompt, prepend_text="", append_text=""):
        # Combine texts
        combined_prompt = ""
        if prepend_text.strip():
            combined_prompt += prepend_text.strip() + " "
        combined_prompt += original_prompt.strip()
        if append_text.strip():
            combined_prompt += " " + append_text.strip()

        combined_prompt = combined_prompt.strip()

        # Only re-encode if the prompt actually changed
        if combined_prompt != original_prompt.strip():
            print(f"[LunaModifyPreprocessedPrompt] Re-encoding modified prompt")
            print(f"[LunaModifyPreprocessedPrompt] Original: {original_prompt}")
            print(f"[LunaModifyPreprocessedPrompt] Modified: {combined_prompt}")

            # Re-encode with CLIP
            from nodes import CLIPTextEncode
            text_encoder = CLIPTextEncode()
            encoded_result = text_encoder.encode(clip, combined_prompt)
            modified_conditioning = encoded_result[0]
        else:
            print(f"[LunaModifyPreprocessedPrompt] No changes detected, using original conditioning")
            modified_conditioning = preprocessed_conditioning

class LunaEmbeddingCache:
    """Global cache for frequently used embeddings"""
    _cache = {}
    _max_cache_size = 100  # Maximum number of cached embeddings
    _cache_hits = 0
    _cache_misses = 0

    @classmethod
    def get(cls, cache_key):
        """Get cached embedding if available"""
        if cache_key in cls._cache:
            cls._cache_hits += 1
            # Move to end (most recently used)
            embedding = cls._cache.pop(cache_key)
            cls._cache[cache_key] = embedding
            return embedding
        cls._cache_misses += 1
        return None

    @classmethod
    def put(cls, cache_key, embedding):
        """Cache an embedding"""
        if len(cls._cache) >= cls._max_cache_size:
            # Remove least recently used
            cls._cache.pop(next(iter(cls._cache)))

        cls._cache[cache_key] = embedding

    @classmethod
    def clear(cls):
        """Clear the cache"""
        cls._cache.clear()
        cls._cache_hits = 0
        cls._cache_misses = 0

    @classmethod
    def get_stats(cls):
        """Get cache statistics"""
        total_requests = cls._cache_hits + cls._cache_misses
        hit_rate = cls._cache_hits / total_requests if total_requests > 0 else 0
        return {
            "cache_size": len(cls._cache),
            "max_cache_size": cls._max_cache_size,
            "cache_hits": cls._cache_hits,
            "cache_misses": cls._cache_misses,
            "hit_rate": hit_rate
        }

class LunaOptimizedPreprocessedLoader:
    CATEGORY = "Luna/Preprocessing"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("positive_conditioning", "negative_conditioning", "original_prompt", "index", "cache_stats")
    FUNCTION = "load_optimized_preprocessed_prompt"

    @classmethod
    def INPUT_TYPES(cls):
        # Get available prompt folders from models/luna_prompts
        luna_prompts_dir = os.path.join(folder_paths.models_dir, "luna_prompts")
        folders = []
        if os.path.exists(luna_prompts_dir):
            folders = [f for f in os.listdir(luna_prompts_dir) if os.path.isdir(os.path.join(luna_prompts_dir, f))]
            folders.sort()

        # Get available negative prompt files
        negative_files = []
        if os.path.exists(luna_prompts_dir):
            for root, dirs, files in os.walk(luna_prompts_dir):
                for file in files:
                    if file.endswith('.safetensors') and not file.startswith('prompt_'):
                        rel_path = os.path.relpath(os.path.join(root, file), luna_prompts_dir)
                        negative_files.append(rel_path)
            negative_files.sort()

        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "tooltip": "Path to the prompt folder containing JSON mappings and safetensors files"}),
                "prompt_key": ("STRING", {"default": "", "tooltip": "Key/name of the prompt to load (from the mappings JSON)"}),
                "negative_prompt_file": (negative_files, {"tooltip": "Select a negative prompt safetensors file"}),
            },
            "optional": {
                "enable_caching": ("BOOLEAN", {"default": True, "tooltip": "Enable embedding caching for faster loading"}),
                "preload_batch": ("INT", {"default": 0, "min": 0, "max": 10, "tooltip": "Number of adjacent prompts to preload into cache (0 = disabled)"}),
            }
        }

    def load_optimized_preprocessed_prompt(self, folder_path, prompt_key, negative_prompt_file, enable_caching=True, preload_batch=0):
        import time
        start_time = time.time()

        # Create cache key
        cache_key = f"{folder_path}:{prompt_key}"

        # Try to get from cache first
        if enable_caching:
            cached_result = LunaEmbeddingCache.get(cache_key)
            if cached_result:
                load_time = time.time() - start_time
                LunaPerformanceMonitor.record_load_time(load_time)
                print(f"[LunaOptimizedPreprocessedLoader] Cache hit for {prompt_key} ({load_time:.3f}s)")
                # Return cached result with cache stats
                cache_stats = LunaEmbeddingCache.get_stats()
                stats_str = f"Cache: {cache_stats['hit_rate']:.1%} hit rate, {cache_stats['cache_size']}/{cache_stats['max_cache_size']} cached"
                return cached_result + (stats_str,)

        # Load from file (cache miss or caching disabled)
        result = self._load_from_file(folder_path, prompt_key, negative_prompt_file)

        load_time = time.time() - start_time
        LunaPerformanceMonitor.record_load_time(load_time)
        print(f"[LunaOptimizedPreprocessedLoader] Loaded {prompt_key} in {load_time:.3f}s")

        # Cache the result if caching is enabled
        if enable_caching and result:
            LunaEmbeddingCache.put(cache_key, result)

        # Preload adjacent prompts if requested
        if preload_batch > 0:
            self._preload_adjacent(folder_path, prompt_key, preload_batch, negative_prompt_file)

        # Add cache stats
        cache_stats = LunaEmbeddingCache.get_stats()
        stats_str = f"Cache: {cache_stats['hit_rate']:.1%} hit rate, {cache_stats['cache_size']}/{cache_stats['max_cache_size']} cached"

        return result + (stats_str,)

    def _load_from_file(self, folder_path, prompt_key, negative_prompt_file):
        """Load prompt from file (same logic as original LunaLoadPreprocessedPrompt)"""
        # Validate inputs
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Find JSON file in the folder
        json_files = [f for f in os.listdir(folder_path) if f.endswith('_mappings.json')]
        if not json_files:
            raise FileNotFoundError(f"No mappings JSON file found in {folder_path}")

        json_path = os.path.join(folder_path, json_files[0])

        # Load mappings
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                mappings = json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading mappings file: {e}")

        # Find the prompt file
        if prompt_key not in mappings:
            available_keys = list(mappings.keys())[:10]
            raise ValueError(f"Prompt key '{prompt_key}' not found in mappings. Available keys: {available_keys}")

        prompt_filepath = mappings[prompt_key]

        # Load the positive prompt safetensors file
        try:
            tensors = load_file(prompt_filepath)
        except Exception as e:
            raise ValueError(f"Error loading positive prompt safetensors file: {e}")

        # Extract data
        if "clip_embeddings" not in tensors:
            raise ValueError("Invalid positive prompt safetensors file: missing 'clip_embeddings' tensor")

        positive_conditioning = tensors["clip_embeddings"]
        original_prompt = tensors.get("original_prompt", prompt_key)
        index = tensors.get("index", -1)

        # Load negative prompt
        negative_conditioning = None
        if negative_prompt_file:
            luna_prompts_dir = os.path.join(folder_paths.models_dir, "luna_prompts")
            negative_filepath = os.path.join(luna_prompts_dir, negative_prompt_file)

            if os.path.exists(negative_filepath):
                try:
                    neg_tensors = load_file(negative_filepath)
                    if "clip_embeddings" in neg_tensors:
                        negative_conditioning = neg_tensors["clip_embeddings"]
                        print(f"[LunaOptimizedPreprocessedLoader] Loaded negative prompt from {negative_filepath}")
                except Exception as e:
                    print(f"[LunaOptimizedPreprocessedLoader] Error loading negative prompt: {e}")

        print(f"[LunaOptimizedPreprocessedLoader] Loaded positive prompt '{prompt_key}' from {prompt_filepath}")
        return (positive_conditioning, negative_conditioning, original_prompt, index)

    def _preload_adjacent(self, folder_path, current_key, preload_count, negative_prompt_file):
        """Preload adjacent prompts into cache"""
        try:
            # Find JSON file
            json_files = [f for f in os.listdir(folder_path) if f.endswith('_mappings.json')]
            if not json_files:
                return

            json_path = os.path.join(folder_path, json_files[0])
            with open(json_path, 'r', encoding='utf-8') as f:
                mappings = json.load(f)

            # Get all keys and find current position
            all_keys = list(mappings.keys())
            try:
                current_index = all_keys.index(current_key)
            except ValueError:
                return

            # Preload adjacent prompts
            for offset in range(1, preload_count + 1):
                # Preload before current
                if current_index - offset >= 0:
                    adjacent_key = all_keys[current_index - offset]
                    cache_key = f"{folder_path}:{adjacent_key}"
                    if LunaEmbeddingCache.get(cache_key) is None:  # Not already cached
                        result = self._load_from_file(folder_path, adjacent_key, negative_prompt_file)
                        if result:
                            LunaEmbeddingCache.put(cache_key, result)
                            print(f"[LunaOptimizedPreprocessedLoader] Preloaded adjacent prompt: {adjacent_key}")

                # Preload after current
                if current_index + offset < len(all_keys):
                    adjacent_key = all_keys[current_index + offset]
                    cache_key = f"{folder_path}:{adjacent_key}"
                    if LunaEmbeddingCache.get(cache_key) is None:  # Not already cached
                        result = self._load_from_file(folder_path, adjacent_key, negative_prompt_file)
                        if result:
                            LunaEmbeddingCache.put(cache_key, result)
                            print(f"[LunaOptimizedPreprocessedLoader] Preloaded adjacent prompt: {adjacent_key}")

        except Exception as e:
            print(f"[LunaOptimizedPreprocessedLoader] Error during preloading: {e}")

class LunaCacheManager:
    CATEGORY = "Luna/Preprocessing"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("cache_info",)
    FUNCTION = "manage_embedding_cache"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["clear_cache", "get_stats", "optimize_cache", "set_max_size"], {"tooltip": "Cache management action"}),
            },
            "optional": {
                "max_cache_size": ("INT", {"default": 100, "min": 10, "max": 1000, "tooltip": "Maximum cache size (only used with set_max_size action)"}),
            }
        }

    def manage_embedding_cache(self, action, max_cache_size=100):
        if action == "clear_cache":
            LunaEmbeddingCache.clear()
            return ("Cache cleared successfully",)

        elif action == "get_stats":
            stats = LunaEmbeddingCache.get_stats()
            info = f"""Embedding Cache Statistics:
Cache Size: {stats['cache_size']}/{stats['max_cache_size']}
Cache Hits: {stats['cache_hits']}
Cache Misses: {stats['cache_misses']}
Hit Rate: {stats['hit_rate']:.1%}
"""
            return (info,)

        elif action == "optimize_cache":
            # Remove least recently used items if cache is full
            current_size = len(LunaEmbeddingCache._cache)
            if current_size > LunaEmbeddingCache._max_cache_size * 0.8:  # If > 80% full
                items_to_remove = int(current_size * 0.2)  # Remove 20%
                for _ in range(items_to_remove):
                    if LunaEmbeddingCache._cache:
                        LunaEmbeddingCache._cache.pop(next(iter(LunaEmbeddingCache._cache)))
                return (f"Optimized cache: removed {items_to_remove} least recently used items",)
            else:
                return ("Cache optimization not needed (cache is not full)",)

        elif action == "set_max_size":
            old_size = LunaEmbeddingCache._max_cache_size
            LunaEmbeddingCache._max_cache_size = max_cache_size
            return (f"Cache max size changed from {old_size} to {max_cache_size}",)

        return ("Unknown action",)

class LunaPerformanceMonitor:
    CATEGORY = "Luna/Preprocessing"
    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT", "INT")
    RETURN_NAMES = ("performance_report", "avg_load_time", "cache_hit_rate", "memory_usage_mb")
    FUNCTION = "monitor_performance"
    OUTPUT_NODE = True

    _load_times = []
    _max_samples = 100

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["get_report", "reset_stats", "analyze_bottlenecks"], {"tooltip": "Performance monitoring action"}),
            },
            "optional": {
                "sample_window": ("INT", {"default": 50, "min": 10, "max": 500, "tooltip": "Number of recent samples to analyze"}),
            }
        }

    def monitor_performance(self, action, sample_window=50):
        import psutil
        import os

        if action == "reset_stats":
            LunaPerformanceMonitor._load_times.clear()
            LunaEmbeddingCache.clear()
            return ("Performance statistics reset", 0.0, 0.0, 0)

        elif action == "get_report":
            # Get cache stats
            cache_stats = LunaEmbeddingCache.get_stats()

            # Calculate average load time
            recent_times = LunaPerformanceMonitor._load_times[-sample_window:]
            avg_load_time = sum(recent_times) / len(recent_times) if recent_times else 0.0

            # Get memory usage
            process = psutil.Process(os.getpid())
            memory_usage_mb = process.memory_info().rss / 1024 / 1024

            # Generate report
            report = f"""Luna Collection Performance Report:

📊 Cache Performance:
   - Hit Rate: {cache_stats['hit_rate']:.1%}
   - Cache Size: {cache_stats['cache_size']}/{cache_stats['max_cache_size']}
   - Total Requests: {cache_stats['cache_hits'] + cache_stats['cache_misses']}

⚡ Loading Performance:
   - Average Load Time: {avg_load_time:.3f}s
   - Samples Analyzed: {len(recent_times)}
   - Fastest Load: {min(recent_times) if recent_times else 0:.3f}s
   - Slowest Load: {max(recent_times) if recent_times else 0:.3f}s

💾 Memory Usage:
   - Current Usage: {memory_usage_mb:.1f} MB
   - Cache Efficiency: {cache_stats['cache_hits'] / max(1, cache_stats['cache_hits'] + cache_stats['cache_misses']) * 100:.1f}% of loads from cache

💡 Recommendations:
   - {'Increase cache size' if cache_stats['hit_rate'] < 0.5 else 'Cache performing well'}
   - {'Consider batch preprocessing' if avg_load_time > 1.0 else 'Loading performance good'}
   - {'Memory usage high - consider quantization' if memory_usage_mb > 2000 else 'Memory usage normal'}
"""

            return (report, avg_load_time, cache_stats['hit_rate'], memory_usage_mb)

        elif action == "analyze_bottlenecks":
            bottlenecks = []

            # Analyze cache performance
            cache_stats = LunaEmbeddingCache.get_stats()
            if cache_stats['hit_rate'] < 0.3:
                bottlenecks.append("Low cache hit rate - consider increasing cache size or preloading")

            # Analyze load times
            recent_times = LunaPerformanceMonitor._load_times[-sample_window:]
            if recent_times:
                avg_time = sum(recent_times) / len(recent_times)
                if avg_time > 2.0:
                    bottlenecks.append("Slow loading times - consider using optimized loader or caching")
                elif avg_time > 0.5:
                    bottlenecks.append("Moderate loading times - batch preprocessing could help")

            # Analyze memory
            process = psutil.Process(os.getpid())
            memory_usage_mb = process.memory_info().rss / 1024 / 1024
            if memory_usage_mb > 3000:
                bottlenecks.append("High memory usage - consider quantized embeddings")

            if not bottlenecks:
                bottlenecks.append("No significant bottlenecks detected - system performing well!")

            analysis = "Performance Analysis:\n" + "\n".join(f"• {b}" for b in bottlenecks)
            return (analysis, 0.0, 0.0, memory_usage_mb)

        return ("Unknown action", 0.0, 0.0, 0)

    @classmethod
    def record_load_time(cls, load_time):
        """Record a load time for performance monitoring"""
        cls._load_times.append(load_time)
        if len(cls._load_times) > cls._max_samples:
            cls._load_times.pop(0)

class LunaWildcardPromptGenerator:
    CATEGORY = "Luna/Preprocessing"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_list_path",)
    FUNCTION = "generate_wildcard_prompts"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        # Get available wildcard files
        wildcards_dir = os.path.join(folder_paths.models_dir, "Wildcards")
        wildcard_files = []

        if os.path.exists(wildcards_dir):
            for root, dirs, files in os.walk(wildcards_dir):
                for file in files:
                    if file.endswith('.txt'):
                        # Get relative path from wildcards directory
                        rel_path = os.path.relpath(os.path.join(root, file), wildcards_dir)
                        wildcard_files.append(rel_path)
            wildcard_files.sort()

        return {
            "required": {
                "wildcard_pattern": ("STRING", {
                    "multiline": True,
                    "default": "__location__, __subject__, __hair/color__, __eyes/color__, __clothing__, __pose__, __accessory__",
                    "tooltip": "Wildcard pattern using __wildcard__ syntax. Separate multiple wildcards with commas."
                }),
                "num_variations": ("INT", {
                    "default": 1000,
                    "min": 1,
                    "max": 10000,
                    "tooltip": "Number of random prompt variations to generate"
                }),
                "output_filename": ("STRING", {
                    "default": "prompt_list.txt",
                    "tooltip": "Name of the output file (will be saved to output/luna_prompts)"
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**31 - 1,
                    "tooltip": "Random seed for reproducible generation (0 = random)"
                }),
                "custom_wildcards_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Custom wildcards directory path (leave empty to use models/Wildcards)"
                }),
            }
        }

    def _resolve_wildcard(self, wildcard_name, wildcards_dir):
        """Resolve a wildcard to its file path and read its contents"""
        # Remove __ from start and end if present
        if wildcard_name.startswith('__') and wildcard_name.endswith('__'):
            wildcard_path = wildcard_name[2:-2]
        else:
            wildcard_path = wildcard_name

        # Handle complex paths with backslashes and nested wildcards
        if '\\' in wildcard_path:
            # Split by backslash and handle each part
            path_parts = wildcard_path.split('\\')
            resolved_parts = []

            for part in path_parts:
                if '__' in part and '__' in part:  # Contains nested wildcards
                    # Recursively resolve nested wildcards
                    nested_wildcards = self._extract_nested_wildcards(part)
                    if nested_wildcards:
                        # For now, return the original part - full nested resolution would be complex
                        # This handles the basic case
                        resolved_parts.append(part)
                    else:
                        resolved_parts.append(part)
                else:
                    resolved_parts.append(part)

            # Try different path combinations
            possible_paths = [
                os.path.join(wildcards_dir, *resolved_parts) + '.txt',
                os.path.join(wildcards_dir, resolved_parts[0], resolved_parts[1]) + '.txt' if len(resolved_parts) > 1 else None,
            ]

            for file_path in possible_paths:
                if file_path and os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = [line.strip() for line in f.readlines() if line.strip()]
                            return lines
                    except Exception as e:
                        print(f"[LunaWildcardPromptGenerator] Error reading wildcard file {file_path}: {e}")
                        continue

        # Standard wildcard resolution
        file_path = os.path.join(wildcards_dir, f"{wildcard_path}.txt")

        # If direct file doesn't exist, try as directory with subdirectory files
        if not os.path.exists(file_path):
            wildcard_parts = wildcard_path.split('/')
            if len(wildcard_parts) > 1:
                # Try subdirectory structure (e.g., hair/color -> Hair/color.txt)
                subdir_path = os.path.join(wildcards_dir, wildcard_parts[0].title(), f"{wildcard_parts[1]}.txt")
                if os.path.exists(subdir_path):
                    file_path = subdir_path

        if not os.path.exists(file_path):
            print(f"[LunaWildcardPromptGenerator] Warning: Wildcard file not found: {file_path}")
            return [wildcard_name]  # Return the original wildcard as fallback

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                return lines
        except Exception as e:
            print(f"[LunaWildcardPromptGenerator] Error reading wildcard file {file_path}: {e}")
            return [wildcard_name]

    def _extract_nested_wildcards(self, text):
        """Extract nested wildcards from text"""
        import re
        return re.findall(r'__([^__]+)__', text)

    def _parse_wildcard_pattern(self, pattern):
        """Parse wildcard pattern and extract individual wildcards"""
        import re

        # Find all __wildcard__ patterns (including weights)
        wildcards = re.findall(r'__([^__]+)__', pattern)

        # Split by commas but be careful with parentheses
        parts = []
        current_part = ""
        paren_depth = 0

        for char in pattern:
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == ',' and paren_depth == 0:
                if current_part.strip():
                    parts.append(current_part.strip())
                current_part = ""
                continue
            current_part += char

        if current_part.strip():
            parts.append(current_part.strip())

        return wildcards, parts

    def _generate_prompt(self, pattern_parts, wildcard_options, seed_offset=0):
        """Generate a single prompt by replacing wildcards with random choices"""
        import random
        import re

        result_parts = []

        for part in pattern_parts:
            temp_part = part

            # Handle regular wildcards (including weights)
            for wildcard in wildcard_options.keys():
                wildcard_tag = f"__{wildcard}__"
                if wildcard_tag in temp_part:
                    options = wildcard_options[wildcard]
                    if options:
                        # Use seed_offset for reproducible but varied results
                        random.seed(seed_offset)
                        choice = random.choice(options)
                        temp_part = temp_part.replace(wildcard_tag, choice)
                        seed_offset += 1

            result_parts.append(temp_part)

        return ', '.join(result_parts)

    def generate_wildcard_prompts(self, wildcard_pattern, num_variations=1000, output_filename="prompt_list.txt", seed=0, custom_wildcards_dir=""):
        # Set up wildcards directory
        if custom_wildcards_dir and os.path.exists(custom_wildcards_dir):
            wildcards_dir = custom_wildcards_dir
        else:
            wildcards_dir = os.path.join(folder_paths.models_dir, "Wildcards")

        if not os.path.exists(wildcards_dir):
            raise FileNotFoundError(f"Wildcards directory not found: {wildcards_dir}")

        # Parse the wildcard pattern
        wildcards, pattern_parts = self._parse_wildcard_pattern(wildcard_pattern)

        if not wildcards:
            raise ValueError("No wildcards found in pattern. Use __wildcard__ syntax.")

        print(f"[LunaWildcardPromptGenerator] Found wildcards: {wildcards}")
        print(f"[LunaWildcardPromptGenerator] Pattern parts: {pattern_parts}")

        # Resolve all wildcards to their options
        wildcard_options = {}
        for wildcard in wildcards:
            options = self._resolve_wildcard(wildcard, wildcards_dir)
            wildcard_options[wildcard] = options
            print(f"[LunaWildcardPromptGenerator] {wildcard}: {len(options)} options")

        # Set up output directory
        output_dir = os.path.join(folder_paths.get_output_directory(), "luna_prompts")
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, output_filename)

        # Generate prompts
        generated_prompts = []
        import random

        if seed == 0:
            random.seed()  # Random seed
        else:
            random.seed(seed)  # Fixed seed for reproducibility

        print(f"[LunaWildcardPromptGenerator] Generating {num_variations} prompt variations...")

        for i in range(num_variations):
            prompt = self._generate_prompt(pattern_parts, wildcard_options, seed_offset=i)
            generated_prompts.append(prompt)

        # Remove duplicates while preserving order
        unique_prompts = []
        seen = set()
        for prompt in generated_prompts:
            if prompt not in seen:
                unique_prompts.append(prompt)
                seen.add(prompt)

        print(f"[LunaWildcardPromptGenerator] Generated {len(unique_prompts)} unique prompts (removed {len(generated_prompts) - len(unique_prompts)} duplicates)")

        # Save to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for prompt in unique_prompts:
                    f.write(prompt + '\n')

            print(f"[LunaWildcardPromptGenerator] Saved {len(unique_prompts)} prompts to: {output_path}")

        except Exception as e:
            raise ValueError(f"Error saving prompt list: {e}")

        return (output_path,)

class LunaListPreprocessedPrompts:
    CATEGORY = "Luna/Preprocessing"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_keys")
    FUNCTION = "list_preprocessed_prompts"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "tooltip": "Path to the prompt folder containing JSON mappings and safetensors files"}),
            }
        }

    def list_preprocessed_prompts(self, folder_path):
        # Validate inputs
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Find JSON file in the folder
        json_files = [f for f in os.listdir(folder_path) if f.endswith('_mappings.json')]
        if not json_files:
            raise FileNotFoundError(f"No mappings JSON file found in {folder_path}")

        json_path = os.path.join(folder_path, json_files[0])

        # Load mappings
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                mappings = json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading mappings file: {e}")

        # Get all prompt keys
        prompt_keys = list(mappings.keys())
        prompt_keys.sort()  # Sort for consistent ordering

        print(f"[LunaListPreprocessedPrompts] Found {len(prompt_keys)} preprocessed prompts in {folder_path}")

        # Return as a single string with newlines for easy reading
        keys_string = "\n".join(prompt_keys)

        return (keys_string,)

class LunaSaveNegativePrompt:
    CATEGORY = "Luna/Preprocessing"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path")
    FUNCTION = "save_negative_prompt"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "CLIP model for text encoding"}),
                "negative_text": ("STRING", {"multiline": True, "default": "", "tooltip": "Negative prompt text to encode and save"}),
                "filename": ("STRING", {"default": "negative_common", "tooltip": "Filename for the saved safetensors file (without extension)"}),
            }
        }

    def save_negative_prompt(self, clip, negative_text, filename):
        # Create luna_prompts directory if it doesn't exist
        luna_prompts_dir = os.path.join(folder_paths.models_dir, "luna_prompts")
        os.makedirs(luna_prompts_dir, exist_ok=True)

        # Initialize text encoder
        from nodes import CLIPTextEncode
        text_encoder = CLIPTextEncode()

        # Encode the negative prompt
        try:
            encoded_result = text_encoder.encode(clip, negative_text)
            encoded_tensor = encoded_result[0]  # CLIPTextEncode returns a tuple
        except Exception as e:
            raise ValueError(f"Error encoding negative prompt: {e}")

        # Save as safetensors
        filepath = os.path.join(luna_prompts_dir, f"{filename}.safetensors")
        tensors_dict = {
            "clip_embeddings": encoded_tensor,
            "original_prompt": negative_text,
            "type": "negative",
            "created": str(datetime.now())
        }

        try:
            save_file(tensors_dict, filepath)
        except Exception as e:
            raise ValueError(f"Error saving negative prompt: {e}")

        print(f"[LunaSaveNegativePrompt] Saved negative prompt to: {filepath}")
        print(f"[LunaSaveNegativePrompt] Text: {negative_text}")

        return (filepath,)

class LunaSinglePromptProcessor:
    CATEGORY = "Luna/Preprocessing"
    RETURN_TYPES = ("STRING", "CONDITIONING")
    RETURN_NAMES = ("saved_path", "conditioning")
    FUNCTION = "process_single_prompt"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "CLIP model for text encoding"}),
                "prompt_text": ("STRING", {"multiline": True, "default": "", "tooltip": "Prompt text to encode and save (supports embeddings with <embedding:name> syntax)"}),
                "filename": ("STRING", {"default": "negative_prompt", "tooltip": "Filename for the saved safetensors file (without extension)"}),
            },
            "optional": {
                "overwrite_existing": ("BOOLEAN", {"default": True, "tooltip": "Overwrite existing file if it exists"}),
            }
        }

    def process_single_prompt(self, clip, prompt_text, filename="negative_prompt", overwrite_existing=True):
        # Create output\luna_prompts directory if it doesn't exist
        output_dir = os.path.join(folder_paths.get_output_directory(), "luna_prompts")
        os.makedirs(output_dir, exist_ok=True)

        # Initialize text encoder
        from nodes import CLIPTextEncode
        text_encoder = CLIPTextEncode()

        # Encode the prompt
        try:
            encoded_result = text_encoder.encode(clip, prompt_text)
            encoded_tensor = encoded_result[0]  # CLIPTextEncode returns a tuple
        except Exception as e:
            raise ValueError(f"Error encoding prompt: {e}")

        # Check if file exists and handle overwrite
        filepath = os.path.join(output_dir, f"{filename}.safetensors")
        if os.path.exists(filepath) and not overwrite_existing:
            print(f"[LunaSinglePromptProcessor] File already exists and overwrite is disabled: {filepath}")
            # Still return the existing file's conditioning if we can load it
            try:
                existing_tensors = load_file(filepath)
                if "clip_embeddings" in existing_tensors:
                    return (filepath, existing_tensors["clip_embeddings"])
            except Exception as e:
                print(f"[LunaSinglePromptProcessor] Error loading existing file: {e}")

        # Save as safetensors
        tensors_dict = {
            "clip_embeddings": encoded_tensor,
            "original_prompt": prompt_text,
            "type": "single_prompt",
            "created": datetime.now().isoformat(),
            "filename": filename
        }

        try:
            save_file(tensors_dict, filepath)
        except Exception as e:
            raise ValueError(f"Error saving prompt: {e}")

        print(f"[LunaSinglePromptProcessor] Saved prompt to: {filepath}")
        print(f"[LunaSinglePromptProcessor] Text: {prompt_text}")

        return (filepath, encoded_tensor)

NODE_CLASS_MAPPINGS = {
    "LunaSelectPromptFolder": LunaSelectPromptFolder,
    "LunaLoadPreprocessedPrompt": LunaLoadPreprocessedPrompt,
    "LunaListPreprocessedPrompts": LunaListPreprocessedPrompts,
    "LunaSaveNegativePrompt": LunaSaveNegativePrompt,
    "LunaSinglePromptProcessor": LunaSinglePromptProcessor,
    "LunaModifyPreprocessedPrompt": LunaModifyPreprocessedPrompt,
    "LunaWildcardPromptGenerator": LunaWildcardPromptGenerator,
    "LunaOptimizedPreprocessedLoader": LunaOptimizedPreprocessedLoader,
    "LunaCacheManager": LunaCacheManager,
    "LunaPerformanceMonitor": LunaPerformanceMonitor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaSelectPromptFolder": "Luna Select Prompt Folder",
    "LunaLoadPreprocessedPrompt": "Luna Load Preprocessed Prompt",
    "LunaListPreprocessedPrompts": "Luna List Preprocessed Prompts",
    "LunaSaveNegativePrompt": "Luna Save Negative Prompt",
    "LunaSinglePromptProcessor": "Luna Single Prompt Processor",
    "LunaModifyPreprocessedPrompt": "Luna Modify Preprocessed Prompt",
    "LunaWildcardPromptGenerator": "Luna Wildcard Prompt Generator",
    "LunaOptimizedPreprocessedLoader": "Luna Optimized Preprocessed Loader",
    "LunaCacheManager": "Luna Cache Manager",
    "LunaPerformanceMonitor": "Luna Performance Monitor",
}