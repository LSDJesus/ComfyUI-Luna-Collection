import torch
import numpy as np
import sys
import os

# Add the parent directory to sys.path to enable relative imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Luna validation system
validation_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "validation")
if validation_path not in sys.path:
    sys.path.insert(0, validation_path)

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

# Import utils modules directly
utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils")
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

try:
    from trt_engine import Engine  # type: ignore
except ImportError:
    print("Luna Detailer: trt_engine not available")
    Engine = None

try:
    from segs import SEG  # type: ignore
    import segs as core  # type: ignore
except ImportError:
    print("Luna Detailer: segs not available")
    SEG = None
    core = None
import nodes
import folder_paths


# Global instance for the TensorRT engine (similar to your original Mediapipe_Engine)
TENSORRT_ENGINE_INSTANCE = None

class Luna_Detailer:
    CATEGORY = "Luna/Detailing"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "engine_path": ("STRING", {"default": "", "tooltip": "Path to the TensorRT .engine file for the main generation engine. Can be in models/tensorrt/ or models/checkpoints/"}),
                "bbox_engine_path": ("STRING", {"default": "", "tooltip": "Path to the TensorRT .engine file for bbox detection. Leave empty to use standard detector. Can be in models/tensorrt/"}),
                "bbox_detector": ("BBOX_DETECTOR",),
                "guide_size": ("FLOAT", {"default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
                "max_size": ("FLOAT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                "noise_mask": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "force_inpaint": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bbox_dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
                "bbox_crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1}),
                "sam_detection_hint": (["center-1", "horizontal-2", "vertical-2", "rect-4", "diamond-4", "mask-area", "mask-points", "mask-point-bbox", "none"],),
                "sam_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "sam_threshold": ("FLOAT", {"default": 0.93, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sam_bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "sam_mask_hint_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sam_mask_hint_use_negative": (["False", "Small", "Outter"],),
                "drop_size": ("INT", {"min": 1, "max": nodes.MAX_RESOLUTION, "step": 1, "default": 10}),
            },
            "optional": {
                "luna_pipe": ("LUNA_PIPE",),
                "model_opt": ("MODEL",),
                "clip_opt": ("CLIP",),
                "vae_opt": ("VAE",),
                "positive_opt": ("CONDITIONING",),
                "negative_opt": ("CONDITIONING",),
                "seed_opt": ("INT", {"min": 0, "max": 0xffffffffffffffff}),
                "sampler_name_opt": ("STRING",),
                "scheduler_opt": ("STRING",),
                "sam_model_opt": ("SAM_MODEL",),
                "segm_detector_opt": ("SEGM_DETECTOR",),
                "height_min": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64, "tooltip": "Min height for TensorRT engine (512/1536/1024)."}),
                "height_max": ("INT", {"default": 1536, "min": 256, "max": 4096, "step": 64, "tooltip": "Max height for TensorRT engine (512/1536/1024)."}),
                "width_min": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64, "tooltip": "Min width for TensorRT engine (512/1536/1024)."}),
                "width_max": ("INT", {"default": 1536, "min": 256, "max": 4096, "step": 64, "tooltip": "Max width for TensorRT engine (512/1536/1024)."}),
                "resize_to_opt": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64, "tooltip": "Optimal size to resize cropped regions to (e.g., 1024 for square). Set to 0 to disable resizing to opt and only clamp to min/max."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "IMAGE", "LUNA_PIPE")
    RETURN_NAMES = ("image", "cropped_refined", "cropped_enhanced_alpha", "mask", "cnet_images", "luna_pipe")
    OUTPUT_IS_LIST = (False, True, True, False, True, False)
    FUNCTION = "doit"
    CATEGORY = "Luna/Detailing"

    def __init__(self):
        self.engine = None

    def load_engine(self, engine_path):
        global TENSORRT_ENGINE_INSTANCE
        if Engine is None:
            raise ImportError("Engine is not available. Please ensure trt_engine.py is properly installed.")
        if TENSORRT_ENGINE_INSTANCE is None or TENSORRT_ENGINE_INSTANCE.engine_path != engine_path:
            TENSORRT_ENGINE_INSTANCE = Engine(engine_path)
            TENSORRT_ENGINE_INSTANCE.load()
            TENSORRT_ENGINE_INSTANCE.activate()
        self.engine = TENSORRT_ENGINE_INSTANCE

    @staticmethod
    def resize_to_engine_bounds(cropped_image_tensor, mask_tensor, height_min, height_max, width_min, width_max, resize_to_opt):
        """
        Resize the cropped image and mask to fit within TensorRT engine bounds.
        - Clamp to min/max if outside range.
        - Resize to opt size if specified (for performance).
        """
        h, w = cropped_image_tensor.shape[2], cropped_image_tensor.shape[3]
        
        # Clamp to min/max
        h = max(height_min, min(height_max, h))
        w = max(width_min, min(width_max, w))
        
        # Resize to opt if enabled
        if resize_to_opt > 0:
            h, w = resize_to_opt, resize_to_opt
        
        # Resize image (bilinear) and mask (nearest)
        cropped_image_resized = torch.nn.functional.interpolate(cropped_image_tensor, size=(h, w), mode="bilinear", align_corners=False)
        mask_resized = torch.nn.functional.interpolate(mask_tensor, size=(h, w), mode="nearest")
        
        return cropped_image_resized, mask_resized

    @staticmethod
    def enhance_face(image, engine, bbox_engine_path, guide_size, guide_size_for_bbox, max_size, seed, feather, noise_mask, force_inpaint,
                     bbox_threshold, bbox_dilation, bbox_crop_factor,
                     sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                     sam_mask_hint_use_negative, drop_size,
                     bbox_detector, segm_detector=None, sam_model_opt=None,
                     height_min=512, height_max=1536, width_min=512, width_max=1536, resize_to_opt=1024):

        if core is None:
            raise ImportError("impact_core is not available. Please ensure impact_core.py is properly installed.")

        # Detection logic (copied from FaceDetailer)
        # Note: For small models like YOLO face detection, TensorRT may provide minimal speed benefit
        # due to fast native inference. Consider using standard detector unless batch processing many images.
        if bbox_engine_path:
            # Future: Implement TensorRT bbox detection here
            # For now, fall back to standard detector
            pass

        bbox_detector.setAux('face')
        segs = bbox_detector.detect(image, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size)
        bbox_detector.setAux(None)

        if sam_model_opt is not None:
            sam_mask = core.make_sam_mask(sam_model_opt, segs, image, sam_detection_hint, sam_dilation,
                                          sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                                          sam_mask_hint_use_negative,)
            segs = core.segs_bitwise_and_mask(segs, sam_mask)

        elif segm_detector is not None:
            segm_segs = segm_detector.detect(image, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size)
            if (hasattr(segm_detector, 'override_bbox_by_segm') and segm_detector.override_bbox_by_segm):
                segs = segm_segs
            else:
                segm_mask = core.segs_to_combined_mask(segm_segs)
                segs = core.segs_bitwise_and_mask(segs, segm_mask)

        if len(segs[1]) == 0:
            return image, [], [], core.segs_to_combined_mask(segs), []

        enhanced_image = image.clone()
        cropped_enhanced = []
        cropped_enhanced_alpha = []
        cnet_pil_list = []

        segs = core.segs_scale_match(segs, image.shape)
        for seg in segs[1]:
            cropped_image = core.crop_ndarray4(image.cpu().numpy(), seg.crop_region)
            cropped_image_tensor = torch.from_numpy(cropped_image).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0  # [1, 3, H, W]

            mask = torch.from_numpy(seg.cropped_mask).unsqueeze(0).unsqueeze(0).float() / 255.0  # [1, 1, H, W]

            # Resize to fit TensorRT engine bounds
            cropped_image_tensor, mask = Luna_Detailer.resize_to_engine_bounds(
                cropped_image_tensor, mask, height_min, height_max, width_min, width_max, resize_to_opt
            )

            # Prepare feed_dict for TensorRT inference
            feed_dict = {
                'input_image': cropped_image_tensor.cuda(),  # Assume engine expects CUDA tensors
                'mask': mask.cuda()
            }

            # Run TensorRT inference
            with torch.cuda.stream(torch.cuda.current_stream()) as stream:
                results = engine.infer(feed_dict, stream)

            # Get output (assume 'output_image' key)
            enhanced_cropped = results['output_image'].cpu().permute(0, 2, 3, 1).squeeze(0).numpy() * 255.0  # [H, W, 3]
            enhanced_cropped = np.clip(enhanced_cropped, 0, 255).astype(np.uint8)

            # Paste back (simplified; adjust for mask blending if needed)
            enhanced_image_np = enhanced_image.cpu().numpy()
            core.paste_ndarray4(enhanced_image_np, enhanced_cropped, seg.crop_region)
            enhanced_image = torch.from_numpy(enhanced_image_np).to(image.device)

            # Handle outputs (similar to FaceDetailer)
            enhanced_image_alpha = torch.from_numpy(enhanced_cropped).float() / 255.0
            mask_resized = mask.squeeze(0)
            core.tensor_putalpha(enhanced_image_alpha, mask_resized)
            cropped_enhanced_alpha.append(enhanced_image_alpha)

            cropped_enhanced.append(torch.from_numpy(enhanced_cropped).float() / 255.0)

        mask_combined = core.segs_to_combined_mask(segs)
        return enhanced_image, cropped_enhanced, cropped_enhanced_alpha, mask_combined, cnet_pil_list

    @conditional_validate('guide_size', min_value=64, max_value=nodes.MAX_RESOLUTION)
    @conditional_validate('max_size', min_value=64, max_value=nodes.MAX_RESOLUTION)
    @conditional_validate('seed', min_value=0, max_value=0xffffffffffffffff)
    @conditional_validate('feather', min_value=0, max_value=100)
    @conditional_validate('bbox_threshold', min_value=0.0, max_value=1.0)
    @conditional_validate('sam_threshold', min_value=0.0, max_value=1.0)
    @conditional_validate('height_min', min_value=64, max_value=nodes.MAX_RESOLUTION)
    @conditional_validate('height_max', min_value=64, max_value=nodes.MAX_RESOLUTION)
    @conditional_validate('width_min', min_value=64, max_value=nodes.MAX_RESOLUTION)
    @conditional_validate('width_max', min_value=64, max_value=nodes.MAX_RESOLUTION)
    @conditional_validate('resize_to_opt', min_value=0, max_value=nodes.MAX_RESOLUTION)
    def doit(self, image, engine_path, bbox_engine_path, bbox_detector, guide_size, guide_size_for, max_size, seed, feather, noise_mask, force_inpaint,
             bbox_threshold, bbox_dilation, bbox_crop_factor,
             sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
             sam_mask_hint_use_negative, drop_size, luna_pipe=None, model_opt=None, clip_opt=None, vae_opt=None,
             positive_opt=None, negative_opt=None, seed_opt=None, sampler_name_opt=None, scheduler_opt=None,
             sam_model_opt=None, segm_detector_opt=None,
             height_min=512, height_max=1536, width_min=512, width_max=1536, resize_to_opt=1024):

        if not engine_path:
            raise ValueError("TensorRT engine path is required.")

        self.load_engine(engine_path)

        # Extract values from luna_pipe or use individual inputs
        if luna_pipe is not None:
            # Unpack luna_pipe: (model, clip, vae, positive, negative, seed, sampler_name, scheduler)
            pipe_model, pipe_clip, pipe_vae, pipe_positive, pipe_negative, pipe_seed, pipe_sampler_name, pipe_scheduler = luna_pipe
            
            model = pipe_model if model_opt is None else model_opt
            clip = pipe_clip if clip_opt is None else clip_opt
            vae = pipe_vae if vae_opt is None else vae_opt
            positive = pipe_positive if positive_opt is None else positive_opt
            negative = pipe_negative if negative_opt is None else negative_opt
            final_seed = pipe_seed if seed_opt is None else seed_opt
            sampler_name = pipe_sampler_name if sampler_name_opt is None else sampler_name_opt
            scheduler = pipe_scheduler if scheduler_opt is None else scheduler_opt
        else:
            # Fallback if no pipe is provided
            model = model_opt
            clip = clip_opt
            vae = vae_opt
            positive = positive_opt
            negative = negative_opt
            final_seed = seed if seed_opt is None else seed_opt
            sampler_name = sampler_name_opt
            scheduler = scheduler_opt

        if len(image) > 1:
            print("[Luna_detailer] Warning: Batch processing enabled; processing all images in batch.")

        result_img = None
        result_cropped_enhanced = []
        result_cropped_enhanced_alpha = []
        result_cnet_images = []
        final_mask = None

        for i, single_image in enumerate(image):
            enhanced_img, cropped_enhanced, cropped_enhanced_alpha, mask, cnet_pil_list = self.enhance_face(
                single_image.unsqueeze(0), self.engine, bbox_engine_path, guide_size, guide_size_for, max_size, final_seed + i, feather, noise_mask, force_inpaint,
                bbox_threshold, bbox_dilation, bbox_crop_factor,
                sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                sam_mask_hint_use_negative, drop_size, bbox_detector, segm_detector_opt, sam_model_opt,
                height_min, height_max, width_min, width_max, resize_to_opt)

            result_img = torch.cat((result_img, enhanced_img), dim=0) if result_img is not None else enhanced_img
            result_cropped_enhanced.extend(cropped_enhanced)
            result_cropped_enhanced_alpha.extend(cropped_enhanced_alpha)
            result_cnet_images.extend(cnet_pil_list)
            final_mask = mask  # Update with the latest mask

        # Handle case where no images were processed
        if result_img is None:
            raise ValueError("No images were processed. Please check your input.")

        # Create luna_pipe for chaining (contains shared parameters only)
        # Format: (model, clip, vae, positive, negative, seed, sampler_name, scheduler)
        luna_pipe = (model, clip, vae, positive, negative, final_seed, sampler_name, scheduler)

        return result_img, result_cropped_enhanced, result_cropped_enhanced_alpha, final_mask, result_cnet_images, luna_pipe



NODE_CLASS_MAPPINGS = {
    "Luna_Detailer": Luna_Detailer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Luna_Detailer": "Luna Detailer",
}

