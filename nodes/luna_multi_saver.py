import os
import json
import threading
import time
from datetime import datetime
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
import folder_paths
from nodes import SaveImage
import comfy.utils
from comfy.cli_args import args

try:
    import piexif
    import piexif.helper
    HAS_PIEXIF = True
except ImportError:
    piexif = None
    HAS_PIEXIF = False

class LunaMultiSaver:
    CATEGORY = "Luna/Image"
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "save_path": ("STRING", {"default": "", "tooltip": "Custom save path relative to output directory (e.g., 'prompt_list/test'). Leave empty to use model_name based paths."}),
                "save_mode": (["parallel", "sequential"], {"default": "parallel", "tooltip": "Parallel saves asynchronously, sequential saves one by one"}),
                "quality_check": ("BOOLEAN", {"default": False, "label_on": "Enable", "label_off": "Disable", "tooltip": "Enable quality-based conditional saving"}),
                "min_quality_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Minimum quality score required for saving"}),
            },
            "optional": {
                "model_name": ("STRING", {"tooltip": "Model name from LunaLoadParameters (used for folder structure)"}),
                "image_1": ("IMAGE", {"tooltip": "First image (required)"}),
                "image_2": ("IMAGE", {"tooltip": "Second image (optional)"}),
                "image_3": ("IMAGE", {"tooltip": "Third image (optional)"}),
                "image_4": ("IMAGE", {"tooltip": "Fourth image (optional)"}),
                "image_5": ("IMAGE", {"tooltip": "Fifth image (optional)"}),
                "affix_1": ("STRING", {"default": "IMAGE1", "tooltip": "Affix name for image_1"}),
                "affix_2": ("STRING", {"default": "IMAGE2", "tooltip": "Affix name for image_2"}),
                "affix_3": ("STRING", {"default": "IMAGE3", "tooltip": "Affix name for image_3"}),
                "affix_4": ("STRING", {"default": "IMAGE4", "tooltip": "Affix name for image_4"}),
                "affix_5": ("STRING", {"default": "IMAGE5", "tooltip": "Affix name for image_5"}),
                "subdir_1": ("BOOLEAN", {"default": True, "label_on": "Subdir", "label_off": "Root", "tooltip": "Save image_1 to affix subdirectory"}),
                "subdir_2": ("BOOLEAN", {"default": True, "label_on": "Subdir", "label_off": "Root", "tooltip": "Save image_2 to affix subdirectory"}),
                "subdir_3": ("BOOLEAN", {"default": True, "label_on": "Subdir", "label_off": "Root", "tooltip": "Save image_3 to affix subdirectory"}),
                "subdir_4": ("BOOLEAN", {"default": True, "label_on": "Subdir", "label_off": "Root", "tooltip": "Save image_4 to affix subdirectory"}),
                "subdir_5": ("BOOLEAN", {"default": True, "label_on": "Subdir", "label_off": "Root", "tooltip": "Save image_5 to affix subdirectory"}),
                "extension": (["png", "webp"], {"default": "png", "tooltip": "File extension/format to save images as"}),
                "lossless_webp": ("BOOLEAN", {"default": True, "label_on": "Lossless", "label_off": "Lossy", "tooltip": "Use lossless compression for WebP files"}),
                "quality_webp": ("INT", {"default": 95, "min": 1, "max": 100, "tooltip": "Quality setting for lossy WebP (1-100, higher = better quality, ignored for lossless WebP)"}),
                "embed_workflow": ("BOOLEAN", {"default": True, "label_on": "Embed", "label_off": "Skip", "tooltip": "Embed workflow metadata in saved images"}),
                "filename": ("STRING", {"default": "", "tooltip": "Filename from LunaTextProcessor (without extension)"}),
                "filename_index": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "Index from LunaTextProcessor"}),
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

    def quality_check_image(self, image, threshold):
        """Simple quality check based on image variance"""
        # Convert to grayscale and calculate variance as a proxy for quality
        if len(image.shape) == 4:  # Batch of images
            image = image[0]  # Take first image

        # Convert to grayscale-like using luminance
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        variance = np.var(gray)

        # Normalize variance to 0-1 range (rough approximation)
        quality_score = min(1.0, variance / 1000.0)
        return quality_score >= threshold

    def save_single_image(self, image, affix_name, use_subdir, model_name, custom_path, custom_metadata, metadata, prompt, extra_pnginfo, batch_timestamp, batch_counter=0, filename="", filename_index=0, extension="png", lossless_webp=True, quality_webp=95, embed_workflow=True):
        """Save a single image with custom filename format and folder structure"""
        try:
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")

            # Determine save path based on model name and subdirectory setting
            if custom_path:
                # Use custom path
                if use_subdir and model_name:
                    save_dir = os.path.join(self.output_dir, custom_path, model_name, affix_name)
                else:
                    save_dir = os.path.join(self.output_dir, custom_path)
            elif model_name:
                # Use model-based structure
                if use_subdir:
                    save_dir = os.path.join(self.output_dir, model_name, affix_name)
                else:
                    save_dir = os.path.join(self.output_dir, model_name)
            else:
                # Fallback to default
                save_dir = self.output_dir

            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)

            results = []
            for (batch_number, img) in enumerate(image):
                i = 255. * img.cpu().numpy()
                pil_img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

                # Generate filename: YYYY_MM_DD_HHMMSS_MODELNAME_AFFIX[_FILENAME][_INDEX][_BATCH#].ext
                model_part = model_name if model_name else "UnknownModel"
                filename_part = f"_{filename}" if filename else ""
                index_part = f"_{filename_index}" if filename_index > 0 else ""
                batch_part = f"_{batch_number}" if len(image) > 1 else ""
                final_filename = f"{batch_timestamp}_{model_part}_{affix_name}{filename_part}{index_part}{batch_part}.{extension}"

                file_path = os.path.join(save_dir, final_filename)

                # Handle metadata based on format
                if extension == 'png':
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
                        png_metadata.add_text("model_name", model_name or "Unknown")
                        png_metadata.add_text("save_timestamp", batch_timestamp)
                        png_metadata.add_text("batch_index", str(batch_number))

                    pil_img.save(file_path, pnginfo=png_metadata, compress_level=4)

                else:
                    # JPEG/WEBP format - use EXIF metadata
                    if embed_workflow and HAS_PIEXIF and piexif is not None and not args.disable_metadata:
                        # Prepare metadata for EXIF
                        pnginfo_json = {}
                        prompt_json = {}
                        if extra_pnginfo is not None:
                            pnginfo_json = {piexif.ImageIFD.Make - i: f"{k}:{json.dumps(v, separators=(',', ':'))}" for i, (k, v) in enumerate(extra_pnginfo.items())}
                        if prompt is not None:
                            prompt_json = {piexif.ImageIFD.Model: f"prompt:{json.dumps(prompt, separators=(',', ':'))}"}

                        # Create EXIF data
                        exif_dict = ({
                            "0th": pnginfo_json | prompt_json
                            } if pnginfo_json or prompt_json else {}) | ({
                            "Exif": {
                                piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(
                                    f"Image Type: {affix_name}, Model: {model_name or 'Unknown'}, Timestamp: {batch_timestamp}, Batch: {batch_number}" +
                                    (f", Generation Metadata: {json.dumps(metadata, separators=(',', ':'))}" if metadata is not None else "") +
                                    (f", Custom: {custom_metadata}" if custom_metadata and metadata is None else ""),
                                    encoding="unicode"
                                )
                            },
                        } if metadata is not None or custom_metadata or True else {})

                        exif_bytes = piexif.dump(exif_dict)
                        pil_img.save(file_path, exif=exif_bytes, quality=quality_webp, lossless=lossless_webp if extension == 'webp' else None)
                    else:
                        # Save without metadata
                        pil_img.save(file_path, quality=quality_webp, lossless=lossless_webp if extension == 'webp' else None)

                results.append({
                    "filename": final_filename,
                    "subfolder": os.path.relpath(save_dir, self.output_dir),
                    "type": "output"
                })

            return results
        except Exception as e:
            print(f"[LunaMultiSaver] Error saving {affix_name} image: {e}")
            return []

    def save_images_parallel(self, images_data, model_name, custom_path, custom_metadata, metadata, prompt, extra_pnginfo, batch_timestamp, filename="", filename_index=0, extension="png", lossless_webp=True, quality_webp=95, embed_workflow=True):
        """Save images in parallel threads"""
        self.save_results = []
        self.save_threads = []

        def save_worker(image_data, idx):
            image, affix_name, use_subdir = image_data
            if image is not None:
                results = self.save_single_image(image, affix_name, use_subdir, model_name, custom_path, custom_metadata, metadata, prompt, extra_pnginfo, batch_timestamp, idx, filename, filename_index, extension, lossless_webp, quality_webp, embed_workflow)
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

    def save_images_sequential(self, images_data, model_name, custom_path, custom_metadata, metadata, prompt, extra_pnginfo, batch_timestamp, filename="", filename_index=0, extension="png", lossless_webp=True, quality_webp=95, embed_workflow=True):
        """Save images sequentially"""
        results = []
        for idx, (image, affix_name, use_subdir) in enumerate(images_data):
            if image is not None:
                batch_results = self.save_single_image(image, affix_name, use_subdir, model_name, custom_path, custom_metadata, metadata, prompt, extra_pnginfo, batch_timestamp, idx, filename, filename_index, extension, lossless_webp, quality_webp, embed_workflow)
                results.extend(batch_results)

        return results

    def save_images(self, save_path, save_mode, quality_check, min_quality_threshold,
                   model_name=None, image_1=None, image_2=None, image_3=None, image_4=None, image_5=None,
                   affix_1="IMAGE1", affix_2="IMAGE2", affix_3="IMAGE3", affix_4="IMAGE4", affix_5="IMAGE5",
                   subdir_1=True, subdir_2=True, subdir_3=True, subdir_4=True, subdir_5=True,
                   extension="png", lossless_webp=True, quality_webp=95, embed_workflow=True,
                   filename="", filename_index=0, custom_metadata="", metadata=None, prompt=None, extra_pnginfo=None):

        # Generate single timestamp for all images in this batch
        batch_timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")

        # Prepare image data with custom affixes and subdirectory settings
        images_data = [
            (image_1, affix_1, subdir_1),
            (image_2, affix_2, subdir_2),
            (image_3, affix_3, subdir_3),
            (image_4, affix_4, subdir_4),
            (image_5, affix_5, subdir_5),
        ]

        # Filter out None images and apply quality check if enabled
        filtered_images = []
        for image, affix, subdir in images_data:
            if image is not None:
                if quality_check:
                    # Check quality of first image in batch
                    if self.quality_check_image(image, min_quality_threshold):
                        filtered_images.append((image, affix, subdir))
                    else:
                        print(f"[LunaMultiSaver] Skipping {affix} - quality below threshold")
                else:
                    filtered_images.append((image, affix, subdir))

        if not filtered_images:
            print("[LunaMultiSaver] No images to save")
            return {"ui": {"images": []}}

        # Save images based on mode
        if save_mode == "parallel":
            results = self.save_images_parallel(filtered_images, model_name, save_path, custom_metadata, metadata, prompt, extra_pnginfo, batch_timestamp, filename, filename_index, extension, lossless_webp, quality_webp, embed_workflow)
        else:
            results = self.save_images_sequential(filtered_images, model_name, save_path, custom_metadata, metadata, prompt, extra_pnginfo, batch_timestamp, filename, filename_index, extension, lossless_webp, quality_webp, embed_workflow)

        return {"ui": {"images": results}}

NODE_CLASS_MAPPINGS = {
    "LunaMultiSaver": LunaMultiSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaMultiSaver": "Luna Multi Image Saver",
}