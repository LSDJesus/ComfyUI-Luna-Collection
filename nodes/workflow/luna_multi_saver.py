import os
import json
import threading
from datetime import datetime
from typing import Dict, Any
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
import folder_paths
from comfy.cli_args import args

try:
    import piexif
    import piexif.helper
    HAS_PIEXIF = True
except ImportError:
    piexif = None
    HAS_PIEXIF = False

# Try to import daemon client for async saving
try:
    from luna_daemon.client import DaemonClient
    DAEMON_AVAILABLE = True
except ImportError:
    DAEMON_AVAILABLE = False
    DaemonClient = None

class LunaMultiSaver:
    CATEGORY = "Luna/Image"
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    
    FORMATS = ["png", "webp", "jpeg"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "save_path": ("STRING", {"default": "", "tooltip": "Path relative to output dir. Supports: %model_path% (full path like Illustrious/3DCG/model), %model_name% (just model name), %index%, %time:FORMAT% (e.g. %time:YYYY-mm-dd%). Empty = output root."}),
                "filename": ("STRING", {"default": "%time:YYYY_mm_dd.HH.MM.SS%_%model_name%", "tooltip": "Filename template. Supports: %model_name%, %index%, %time:FORMAT% (e.g. %time:YYYY-mm-dd.HH.MM.SS%). Affix is always appended."}),
                "daemon_save": ("BOOLEAN", {"default": False, "label_on": "Async (Daemon)", "label_off": "Blocking", "tooltip": "Use daemon async saving (non-blocking) or local blocking save. Requires daemon to be running."}),
                "save_mode": (["parallel", "sequential"], {"default": "parallel", "tooltip": "Parallel saves asynchronously, sequential saves one by one"}),
                "quality_gate": (["disabled", "variance", "edge_density", "both"], {"default": "disabled", "tooltip": "Quality check mode: variance detects flat/blank images, edge_density detects blurry images, both requires passing both checks"}),
                "min_quality_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Minimum quality score required for saving (0.3 recommended for filtering obviously bad images)"}),
            },
            "optional": {
                "model_name": ("STRING", {"tooltip": "(Deprecated) Model name - now auto-extracted from metadata input. Only needed if not using LunaConfigGateway."}),
                "image_1": ("IMAGE", {"tooltip": "First image"}),
                "image_2": ("IMAGE", {"tooltip": "Second image"}),
                "image_3": ("IMAGE", {"tooltip": "Third image"}),
                "image_4": ("IMAGE", {"tooltip": "Fourth image"}),
                "image_5": ("IMAGE", {"tooltip": "Fifth image"}),
                "affix_1": ("STRING", {"default": "ORIG", "tooltip": "Affix name for image_1"}),
                "affix_2": ("STRING", {"default": "UPSCALED", "tooltip": "Affix name for image_2"}),
                "affix_3": ("STRING", {"default": "DETAILED", "tooltip": "Affix name for image_3"}),
                "affix_4": ("STRING", {"default": "FINAL", "tooltip": "Affix name for image_4"}),
                "affix_5": ("STRING", {"default": "IMAGE5", "tooltip": "Affix name for image_5"}),
                "format_1": (["png", "webp", "jpeg", "no save"], {"default": "webp", "tooltip": "Format for image_1. Select 'no save' to skip saving this image."}),
                "format_2": (["png", "webp", "jpeg", "no save"], {"default": "webp", "tooltip": "Format for image_2. Select 'no save' to skip saving this image."}),
                "format_3": (["png", "webp", "jpeg", "no save"], {"default": "webp", "tooltip": "Format for image_3. Select 'no save' to skip saving this image."}),
                "format_4": (["png", "webp", "jpeg", "no save"], {"default": "webp", "tooltip": "Format for image_4. Select 'no save' to skip saving this image."}),
                "format_5": (["png", "webp", "jpeg", "no save"], {"default": "webp", "tooltip": "Format for image_5. Select 'no save' to skip saving this image."}),
                "subdir_1": ("BOOLEAN", {"default": True, "label_on": "Subdir", "label_off": "Root", "tooltip": "Save image_1 to affix subdirectory"}),
                "subdir_2": ("BOOLEAN", {"default": True, "label_on": "Subdir", "label_off": "Root", "tooltip": "Save image_2 to affix subdirectory"}),
                "subdir_3": ("BOOLEAN", {"default": True, "label_on": "Subdir", "label_off": "Root", "tooltip": "Save image_3 to affix subdirectory"}),
                "subdir_4": ("BOOLEAN", {"default": True, "label_on": "Subdir", "label_off": "Root", "tooltip": "Save image_4 to affix subdirectory"}),
                "subdir_5": ("BOOLEAN", {"default": True, "label_on": "Subdir", "label_off": "Root", "tooltip": "Save image_5 to affix subdirectory"}),
                "png_compression": ("INT", {"default": 4, "min": 0, "max": 9, "tooltip": "PNG compression level (0=fastest/no compression, 9=smallest/slowest). Quality is identical - only affects file size and save speed."}),
                "lossy_quality": ("INT", {"default": 90, "min": 70, "max": 100, "tooltip": "Quality for JPEG and lossy WebP (70-100). Below 70 produces visible artifacts."}),
                "lossless_webp": ("BOOLEAN", {"default": False, "label_on": "Lossless", "label_off": "Lossy", "tooltip": "Use lossless compression for WebP files (ignores lossy_quality setting)"}),
                "embed_workflow": ("BOOLEAN", {"default": True, "label_on": "Embed", "label_off": "Skip", "tooltip": "Embed workflow metadata in saved images"}),
                "filename_index": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "Index value for %index% template variable. Connect from LunaBatchPromptLoader index output."}),
                "custom_metadata": ("STRING", {"multiline": True, "default": "", "tooltip": "Custom metadata added to all images"}),
                "metadata": ("METADATA", {"tooltip": "Metadata from LunaLoadParameters node (overrides custom_metadata)"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.save_threads = []
        self.save_results = []

    def _parse_model_info(self, model_name_raw):
        """
        Parse model name into components.
        Input: 'Illustrious/3DCG/hsUltrahdCG_v60.safetensors' or 'Illustrious\\3DCG\\hsUltrahdCG_v60.safetensors'
        Returns: (model_path='Illustrious/3DCG/hsUltrahdCG_v60', model_name='hsUltrahdCG_v60', model_dir='Illustrious/3DCG')
        """
        if not model_name_raw:
            return '', 'UnknownModel', ''
        
        # Normalize path separators to forward slash
        normalized = model_name_raw.replace('\\', '/')
        
        # Remove trailing slashes
        normalized = normalized.rstrip('/')
        
        # Get basename and remove extension
        basename = os.path.basename(normalized)
        model_name = basename
        for ext in ('.safetensors', '.ckpt', '.pt', '.pth', '.bin', '.gguf'):
            if model_name.lower().endswith(ext):
                model_name = model_name[:-len(ext)]
                break
        
        # Get directory portion (model path without filename)
        dirname = os.path.dirname(normalized)
        
        # Full model path = dir + model_name (without extension)
        if dirname:
            model_path = f"{dirname}/{model_name}"
        else:
            model_path = model_name
        
        return model_path, model_name, dirname

    def _process_time_template(self, template_str, timestamp=None):
        """
        Process %time:FORMAT% templates.
        FORMAT uses: YYYY=year, mm=month, dd=day, HH=hour, MM=minute, SS=second
        Example: %time:YYYY-mm-dd.HH.MM.SS% -> 2025-12-02.23.36.42
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        import re
        pattern = r'%time:([^%]+)%'
        
        def replace_time(match):
            fmt = match.group(1)
            # Convert our format tokens to Python strftime
            fmt = fmt.replace('YYYY', '%Y')
            fmt = fmt.replace('mm', '%m')
            fmt = fmt.replace('dd', '%d')
            fmt = fmt.replace('HH', '%H')
            fmt = fmt.replace('MM', '%M')
            fmt = fmt.replace('SS', '%S')
            return timestamp.strftime(fmt)
        
        return re.sub(pattern, replace_time, template_str)

    def _process_template(self, template_str, model_path, model_name, model_dir, timestamp=None, index=0):
        """
        Process all template variables in a string.
        Supports: %model_path%, %model_name%, %model_dir%, %index%, %time:FORMAT%
        """
        if not template_str:
            return ''
        
        result = template_str
        
        # Process model templates
        result = result.replace('%model_path%', model_path)
        result = result.replace('%model_name%', model_name)
        result = result.replace('%model_dir%', model_dir)
        
        # Process index template
        result = result.replace('%index%', str(index))
        
        # Process time templates
        result = self._process_time_template(result, timestamp)
        
        # Normalize path separators and remove trailing slashes
        result = result.replace('\\', '/')
        result = result.rstrip('/')
        
        return result

    def check_variance(self, image, threshold):
        """Check if image has enough variance (not flat/blank)"""
        if len(image.shape) == 4:
            image = image[0]
        
        # Convert to grayscale using luminance
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        variance = np.var(gray)
        
        # Normalize: typical SD images have variance 2000-8000, blank ~0, solid color ~0
        # Scale so 0.3 threshold catches obviously bad images
        score = min(1.0, variance / 3000.0)
        return score, score >= threshold
    
    def check_edge_density(self, image, threshold):
        """Check if image has enough edge detail (not blurry/smooth)"""
        if len(image.shape) == 4:
            image = image[0]
        
        # Convert to grayscale
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        
        # Simple Sobel-like edge detection
        # Horizontal edges
        gx = np.abs(gray[:-1, :] - gray[1:, :])
        # Vertical edges
        gy = np.abs(gray[:, :-1] - gray[:, 1:])
        
        # Edge density = mean edge magnitude
        edge_mean = (np.mean(gx) + np.mean(gy)) / 2.0
        
        # Normalize: typical detailed images have edge_mean 0.02-0.08, blurry ~0.005
        score = min(1.0, edge_mean / 0.05)
        return score, score >= threshold

    def quality_check_image(self, image, threshold, mode):
        """Check image quality based on selected mode"""
        if mode == "disabled":
            return True, 1.0, "disabled"
        
        if mode == "variance":
            score, passed = self.check_variance(image, threshold)
            return passed, score, "variance"
        
        if mode == "edge_density":
            score, passed = self.check_edge_density(image, threshold)
            return passed, score, "edge_density"
        
        if mode == "both":
            var_score, var_passed = self.check_variance(image, threshold)
            edge_score, edge_passed = self.check_edge_density(image, threshold)
            combined_score = (var_score + edge_score) / 2.0
            return var_passed and edge_passed, combined_score, f"variance={var_score:.2f}, edge={edge_score:.2f}"
        
        return True, 1.0, "unknown"

    def save_single_image(self, image, affix_name, use_subdir, model_name_raw, custom_path, custom_metadata, metadata, prompt, extra_pnginfo, batch_timestamp, batch_counter=0, filename_template="", filename_index=0, extension="png", lossless_webp=False, lossy_quality=90, png_compression=4, embed_workflow=True):
        """Save a single image with custom filename format and folder structure"""
        try:
            # Normalize extension
            ext = extension.lower()
            if ext == "jpeg":
                ext = "jpg"
            
            # Parse timestamp for consistent time across templates
            try:
                timestamp = datetime.strptime(batch_timestamp, "%Y_%m_%d_%H%M%S")
            except:
                timestamp = datetime.now()
            
            # Parse model info from raw model name
            model_path, model_name, model_dir = self._parse_model_info(model_name_raw)
            
            # Process save_path template (index can be used in path too)
            processed_path = self._process_template(custom_path, model_path, model_name, model_dir, timestamp, filename_index)
            
            # Build final save directory
            if processed_path:
                if use_subdir:
                    save_dir = os.path.join(self.output_dir, processed_path, affix_name)
                else:
                    save_dir = os.path.join(self.output_dir, processed_path)
            else:
                # Empty path = output root
                if use_subdir:
                    save_dir = os.path.join(self.output_dir, affix_name)
                else:
                    save_dir = self.output_dir

            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)

            results = []
            print(f"[LunaMultiSaver] Processing {len(image)} image(s) for {affix_name}, format={ext}")
            
            for (batch_number, img) in enumerate(image):
                i = 255. * img.cpu().numpy()
                pil_img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

                # Process filename template
                processed_filename = self._process_template(filename_template, model_path, model_name, model_dir, timestamp, filename_index)
                
                # If no template provided, use default format
                if not processed_filename:
                    processed_filename = f"{batch_timestamp}_{model_name}"
                
                # Build final filename with affix
                # Note: %index% is now handled in template, so we don't append separately
                batch_part = f"_{batch_number}" if len(image) > 1 else ""
                final_filename = f"{processed_filename}_{affix_name}{batch_part}.{ext}"

                file_path = os.path.join(save_dir, final_filename)
                print(f"[LunaMultiSaver] Saving to: {file_path}")

                # Handle metadata based on format
                if ext == 'png':
                    # PNG format - use PngInfo
                    png_metadata = None
                    if embed_workflow and not args.disable_metadata:
                        png_metadata = PngInfo()
                        if prompt is not None:
                            png_metadata.add_text("prompt", json.dumps(prompt))
                        if extra_pnginfo is not None:
                            for x in extra_pnginfo:
                                png_metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                        # Add structured metadata from LunaLoadParameters if available
                        if metadata is not None:
                            png_metadata.add_text("generation_metadata", json.dumps(metadata))
                            # Extract individual fields for easier access
                            for key, value in metadata.items():
                                png_metadata.add_text(f"gen_{key}", str(value))
                        elif custom_metadata:
                            png_metadata.add_text("custom_metadata", custom_metadata)

                        png_metadata.add_text("image_type", affix_name)
                        png_metadata.add_text("model_name", model_name)
                        png_metadata.add_text("model_path", model_path)
                        png_metadata.add_text("save_timestamp", batch_timestamp)
                        png_metadata.add_text("batch_index", str(batch_number))

                    pil_img.save(file_path, pnginfo=png_metadata, compress_level=png_compression)

                elif ext == 'webp':
                    # WebP format
                    # Note: PIL WebP support for EXIF is limited, we embed as XMP or skip
                    save_kwargs: Dict[str, Any] = {
                        'lossless': lossless_webp,
                    }
                    if not lossless_webp:
                        save_kwargs['quality'] = lossy_quality
                    
                    # Try to embed EXIF if piexif is available
                    if embed_workflow and HAS_PIEXIF and piexif is not None and not args.disable_metadata:
                        try:
                            exif_bytes = self._build_exif_metadata(affix_name, model_name, model_path, batch_timestamp, batch_number, metadata, custom_metadata, prompt, extra_pnginfo)
                            save_kwargs['exif'] = exif_bytes
                        except Exception as exif_error:
                            print(f"[LunaMultiSaver] Warning: Could not embed EXIF in WebP: {exif_error}")
                    
                    try:
                        pil_img.save(file_path, 'WEBP', **save_kwargs)
                    except Exception as webp_error:
                        # Fallback: try without exif if it failed
                        print(f"[LunaMultiSaver] WebP save error, retrying without EXIF: {webp_error}")
                        save_kwargs.pop('exif', None)
                        pil_img.save(file_path, 'WEBP', **save_kwargs)

                else:  # jpg/jpeg
                    # JPEG format - convert to RGB if needed (no alpha)
                    if pil_img.mode in ('RGBA', 'LA', 'P'):
                        pil_img = pil_img.convert('RGB')
                    
                    if embed_workflow and HAS_PIEXIF and piexif is not None and not args.disable_metadata:
                        exif_bytes = self._build_exif_metadata(affix_name, model_name, model_path, batch_timestamp, batch_number, metadata, custom_metadata, prompt, extra_pnginfo)
                        pil_img.save(file_path, exif=exif_bytes, quality=lossy_quality)
                    else:
                        pil_img.save(file_path, quality=lossy_quality)

                results.append({
                    "filename": final_filename,
                    "subfolder": os.path.relpath(save_dir, self.output_dir),
                    "type": "output"
                })

            return results
        except Exception as e:
            print(f"[LunaMultiSaver] Error saving {affix_name} image: {e}")
            return []
    
    def _build_exif_metadata(self, affix_name, model_name, model_path, batch_timestamp, batch_number, metadata, custom_metadata, prompt, extra_pnginfo):
        """Build EXIF metadata for JPEG/WebP files"""
        pnginfo_json = {}
        prompt_json = {}
        if extra_pnginfo is not None:
            pnginfo_json = {piexif.ImageIFD.Make - i: f"{k}:{json.dumps(v, separators=(',', ':'))}" for i, (k, v) in enumerate(extra_pnginfo.items())}
        if prompt is not None:
            prompt_json = {piexif.ImageIFD.Model: f"prompt:{json.dumps(prompt, separators=(',', ':'))}"}

        exif_dict = ({
            "0th": pnginfo_json | prompt_json
            } if pnginfo_json or prompt_json else {}) | ({
            "Exif": {
                piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(
                    f"Image Type: {affix_name}, Model: {model_name}, Model Path: {model_path}, Timestamp: {batch_timestamp}, Batch: {batch_number}" +
                    (f", Generation Metadata: {json.dumps(metadata, separators=(',', ':'))}" if metadata is not None else "") +
                    (f", Custom: {custom_metadata}" if custom_metadata and metadata is None else ""),
                    encoding="unicode"
                )
            },
        })

        return piexif.dump(exif_dict)

    def save_images_parallel(self, images_data, model_name_raw, custom_path, custom_metadata, metadata, prompt, extra_pnginfo, batch_timestamp, filename_template="", filename_index=0, lossless_webp=False, lossy_quality=90, png_compression=4, embed_workflow=True):
        """Save images in parallel threads"""
        self.save_results = []
        self.save_threads = []
        results_lock = threading.Lock()

        def save_worker(image_data, idx):
            image, affix_name, use_subdir, fmt = image_data
            if image is not None:
                results = self.save_single_image(image, affix_name, use_subdir, model_name_raw, custom_path, custom_metadata, metadata, prompt, extra_pnginfo, batch_timestamp, idx, filename_template, filename_index, fmt, lossless_webp, lossy_quality, png_compression, embed_workflow)
                with results_lock:
                    self.save_results.extend(results)

        # Start parallel saves
        for idx, image_data in enumerate(images_data):
            if image_data[0] is not None:  # Only save if image exists
                thread = threading.Thread(target=save_worker, args=(image_data, idx))
                thread.daemon = True
                self.save_threads.append(thread)
                thread.start()

        # Wait for all threads to complete
        for thread in self.save_threads:
            thread.join()

        return self.save_results

    def _save_images_via_daemon(self, filtered_images, discarded_images, model_name_raw, save_path,
                                custom_metadata, metadata, prompt, extra_pnginfo, batch_timestamp,
                                filename, filename_index, lossless_webp, lossy_quality, png_compression, embed_workflow,
                                quality_gate, min_quality_threshold):
        """Submit images to daemon for async saving and return immediately"""
        if not DAEMON_AVAILABLE or not DaemonClient:
            error_msg = "[LunaMultiSaver] Daemon save selected but daemon client not available. Is Luna Daemon running? Falling back to local save."
            print(error_msg)
            # Fall back to local parallel save
            return self._fallback_to_local_save(
                filtered_images, discarded_images, model_name_raw, save_path,
                custom_metadata, metadata, prompt, extra_pnginfo, batch_timestamp,
                filename, filename_index, lossless_webp, lossy_quality, png_compression, embed_workflow
            )
        
        try:
            client = DaemonClient()
            
            # Parse timestamp once for consistent time across all images
            try:
                timestamp = datetime.strptime(batch_timestamp, "%Y_%m_%d_%H%M%S")
            except:
                timestamp = datetime.now()
            
            # Parse model info once
            model_path, model_name, model_dir = self._parse_model_info(model_name_raw)
            
            # Process path templates once (same for all images)
            processed_save_path = self._process_template(save_path, model_path, model_name, model_dir, timestamp, filename_index)
            processed_filename_base = self._process_template(filename, model_path, model_name, model_dir, timestamp, filename_index)
            if not processed_filename_base:
                processed_filename_base = f"{batch_timestamp}_{model_name}"
            
            # Collect image data for daemon
            images_for_daemon = []
            for image, affix, subdir, fmt in filtered_images:
                # Convert tensor to numpy
                image_np = image
                if hasattr(image_np, 'cpu'):
                    image_np = image_np.cpu().numpy()
                elif isinstance(image_np, np.ndarray):
                    pass  # Already numpy
                else:
                    image_np = np.array(image_np)
                
                # Handle batch dimension - process each image in the batch
                if image_np.ndim == 4:
                    # Batch of images - process each one
                    for batch_idx in range(image_np.shape[0]):
                        single_image_np = image_np[batch_idx]
                        
                        # Ensure proper range
                        if single_image_np.max() <= 1.0:
                            single_image_np = (single_image_np * 255).astype(np.uint8)
                        else:
                            single_image_np = single_image_np.astype(np.uint8)
                        
                        # Build final directory path
                        if processed_save_path:
                            if subdir:
                                final_save_dir = os.path.join(processed_save_path, affix)
                            else:
                                final_save_dir = processed_save_path
                        else:
                            # Empty path = root
                            if subdir:
                                final_save_dir = affix
                            else:
                                final_save_dir = ""
                        
                        # Build final filename with affix and batch index if needed
                        if image_np.shape[0] > 1:
                            final_filename = f"{processed_filename_base}_{affix}_{batch_idx}"
                        else:
                            final_filename = f"{processed_filename_base}_{affix}"
                        
                        images_for_daemon.append({
                            "image": single_image_np,
                            "save_dir": final_save_dir,
                            "filename": final_filename,
                            "format": fmt
                        })
                else:
                    # Single image (3D tensor)
                    if image_np.max() <= 1.0:
                        image_np = (image_np * 255).astype(np.uint8)
                    else:
                        image_np = image_np.astype(np.uint8)
                    
                    # Build final directory path
                    if processed_save_path:
                        if subdir:
                            final_save_dir = os.path.join(processed_save_path, affix)
                        else:
                            final_save_dir = processed_save_path
                    else:
                        # Empty path = root
                        if subdir:
                            final_save_dir = affix
                        else:
                            final_save_dir = ""
                    
                    # Build final filename with affix
                    final_filename = f"{processed_filename_base}_{affix}"
                    
                    images_for_daemon.append({
                        "image": image_np,
                        "save_dir": final_save_dir,
                        "filename": final_filename,
                        "format": fmt
                    })
            
            # Build daemon request - much simpler now, no templates needed
            save_request = {
                "quality_gate": quality_gate,
                "min_quality_threshold": min_quality_threshold,
                "png_compression": png_compression,
                "lossy_quality": lossy_quality,
                "lossless_webp": lossless_webp,
                "embed_workflow": embed_workflow,
                "custom_metadata": custom_metadata,
                "metadata": metadata or {},
                "prompt": prompt,
                "extra_pnginfo": extra_pnginfo,
                "images": images_for_daemon,
                "output_dir": self.output_dir,
                "model_name": model_name,  # Just for metadata embedding
                "model_path": model_path,  # Just for metadata embedding
                "timestamp": batch_timestamp,  # For metadata embedding
            }
            
            # Submit to daemon
            result = client.submit_async("save_images_async", save_request)
            
            if result.get("success"):
                job_id = result.get("job_id", "unknown")
                num_images = result.get("num_images", len(images_for_daemon))
                msg = f"[LunaMultiSaver] Submitted {num_images} images to daemon (Job ID: {job_id})"
                print(msg)
                return {
                    "ui": {
                        "text": [msg],
                        "images": []
                    }
                }
            else:
                error_msg = f"[LunaMultiSaver] Daemon submission failed: {result.get('error', 'Unknown error')}"
                print(error_msg)
                # Fall back to local save
                return self._fallback_to_local_save(
                    filtered_images, discarded_images, model_name_raw, save_path,
                    custom_metadata, metadata, prompt, extra_pnginfo, batch_timestamp,
                    filename, filename_index, lossless_webp, lossy_quality, png_compression, embed_workflow
                )
        
        except Exception as e:
            error_msg = f"[LunaMultiSaver] Daemon save error: {str(e)}. Falling back to local save."
            print(error_msg)
            # Fall back to local parallel save
            return self._fallback_to_local_save(
                filtered_images, discarded_images, model_name_raw, save_path,
                custom_metadata, metadata, prompt, extra_pnginfo, batch_timestamp,
                filename, filename_index, lossless_webp, lossy_quality, png_compression, embed_workflow
            )
    
    def _fallback_to_local_save(self, filtered_images, discarded_images, model_name_raw, save_path,
                                custom_metadata, metadata, prompt, extra_pnginfo, batch_timestamp,
                                filename, filename_index, lossless_webp, lossy_quality, png_compression, embed_workflow):
        """Fallback to local parallel save (same as normal operation)"""
        all_results = []
        
        # Save passed images
        if filtered_images:
            results = self.save_images_parallel(filtered_images, model_name_raw, save_path, custom_metadata, metadata, prompt, extra_pnginfo, batch_timestamp, filename, filename_index, lossless_webp, lossy_quality, png_compression, embed_workflow)
            all_results.extend(results)
        
        # Save discarded images
        if discarded_images:
            discarded_path = os.path.join(save_path, "_discarded") if save_path else "_discarded"
            discarded_data = [(img, affix, subdir, fmt) for img, affix, subdir, fmt, score, mode in discarded_images]
            discarded_results = self.save_images_parallel(discarded_data, model_name_raw, discarded_path, custom_metadata, metadata, prompt, extra_pnginfo, batch_timestamp, filename, filename_index, lossless_webp, lossy_quality, png_compression, embed_workflow)
            for r in discarded_results:
                r["discarded"] = True
            all_results.extend(discarded_results)
        
        if not all_results:
            return {"ui": {"images": []}}
        
        return {"ui": {"images": all_results}}

    def save_images_sequential(self, images_data, model_name_raw, custom_path, custom_metadata, metadata, prompt, extra_pnginfo, batch_timestamp, filename_template="", filename_index=0, lossless_webp=False, lossy_quality=90, png_compression=4, embed_workflow=True):
        """Save images sequentially"""
        results = []
        for idx, (image, affix_name, use_subdir, fmt) in enumerate(images_data):
            if image is not None:
                batch_results = self.save_single_image(image, affix_name, use_subdir, model_name_raw, custom_path, custom_metadata, metadata, prompt, extra_pnginfo, batch_timestamp, idx, filename_template, filename_index, fmt, lossless_webp, lossy_quality, png_compression, embed_workflow)
                results.extend(batch_results)

        return results

    def save_images(self, save_path, filename, daemon_save, save_mode, quality_gate, min_quality_threshold,
                   model_name=None, model_path_raw=None, image_1=None, image_2=None, image_3=None, image_4=None, image_5=None,
                   affix_1="RAW", affix_2="UPSCALED", affix_3="DETAILED", affix_4="IMAGE4", affix_5="IMAGE5",
                   format_1="png", format_2="png", format_3="png", format_4="png", format_5="png",
                   subdir_1=True, subdir_2=True, subdir_3=True, subdir_4=True, subdir_5=True,
                   png_compression=4, lossy_quality=90, lossless_webp=False, embed_workflow=True,
                   filename_index=0, custom_metadata="", metadata=None, prompt=None, extra_pnginfo=None):

        # Extract model_name from metadata if available, otherwise use input
        # Priority: metadata > model_name input > empty string
        if metadata and isinstance(metadata, dict):
            model_name_raw = metadata.get("model_name", metadata.get("model", ""))
        else:
            model_name_raw = model_name or ""
        
        if not model_name_raw and model_name:
            # Fallback to direct input if metadata doesn't have it
            model_name_raw = model_name

        # Generate single timestamp for all images in this batch
        batch_timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")

        # Prepare image data with custom affixes, subdirectory settings, and formats
        images_data = [
            (image_1, affix_1, subdir_1, format_1),
            (image_2, affix_2, subdir_2, format_2),
            (image_3, affix_3, subdir_3, format_3),
            (image_4, affix_4, subdir_4, format_4),
            (image_5, affix_5, subdir_5, format_5),
        ]

        # Separate images into passed and discarded based on quality check
        filtered_images = []
        discarded_images = []
        
        for image, affix, subdir, fmt in images_data:
            # Skip images with "no save" format
            if fmt == "no save":
                if image is not None:
                    print(f"[LunaMultiSaver] Skipping {affix} - format set to 'no save'")
                continue
            
            if image is not None:
                passed, score, mode_info = self.quality_check_image(image, min_quality_threshold, quality_gate)
                if passed:
                    filtered_images.append((image, affix, subdir, fmt))
                else:
                    # Save to discarded folder instead of skipping entirely
                    discarded_images.append((image, affix, subdir, fmt, score, mode_info))
                    print(f"[LunaMultiSaver] Discarding {affix} - quality check failed ({mode_info}, score={score:.2f}, threshold={min_quality_threshold})")

        all_results = []
        
        # Check if using daemon save
        if daemon_save:
            return self._save_images_via_daemon(
                filtered_images, discarded_images, model_name_raw, save_path,
                custom_metadata, metadata, prompt, extra_pnginfo, batch_timestamp,
                filename, filename_index, lossless_webp, lossy_quality, png_compression, embed_workflow,
                quality_gate, min_quality_threshold
            )
        
        # Save passed images to normal location (local blocking save)
        if filtered_images:
            if save_mode == "parallel":
                results = self.save_images_parallel(filtered_images, model_name_raw, save_path, custom_metadata, metadata, prompt, extra_pnginfo, batch_timestamp, filename, filename_index, lossless_webp, lossy_quality, png_compression, embed_workflow)
            else:
                results = self.save_images_sequential(filtered_images, model_name_raw, save_path, custom_metadata, metadata, prompt, extra_pnginfo, batch_timestamp, filename, filename_index, lossless_webp, lossy_quality, png_compression, embed_workflow)
            all_results.extend(results)
        
        # Save discarded images to _discarded subfolder for review
        if discarded_images:
            discarded_path = os.path.join(save_path, "_discarded") if save_path else "_discarded"
            discarded_data = [(img, affix, subdir, fmt) for img, affix, subdir, fmt, score, mode in discarded_images]
            
            if save_mode == "parallel":
                discarded_results = self.save_images_parallel(discarded_data, model_name_raw, discarded_path, custom_metadata, metadata, prompt, extra_pnginfo, batch_timestamp, filename, filename_index, lossless_webp, lossy_quality, png_compression, embed_workflow)
            else:
                discarded_results = self.save_images_sequential(discarded_data, model_name_raw, discarded_path, custom_metadata, metadata, prompt, extra_pnginfo, batch_timestamp, filename, filename_index, lossless_webp, lossy_quality, png_compression, embed_workflow)
            
            # Mark discarded results for UI distinction (optional)
            for r in discarded_results:
                r["discarded"] = True
            all_results.extend(discarded_results)

        if not all_results:
            print("[LunaMultiSaver] No images to save")
            return {"ui": {"images": []}}

        return {"ui": {"images": all_results}}

NODE_CLASS_MAPPINGS = {
    "LunaMultiSaver": LunaMultiSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaMultiSaver": "Luna Multi Image Saver",
}