import torch
import numpy as np

ENGINE_INSTANCE = None

class Luna_MediaPipe_Segs:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detect_hands": ("BOOLEAN", {"default": False}),
                "detect_face": ("BOOLEAN", {"default": False}),
                "detect_eyes": ("BOOLEAN", {"default": False}),
                "detect_mouth": ("BOOLEAN", {"default": False}),
                "detect_feet": ("BOOLEAN", {"default": False}),
                "detect_torso": ("BOOLEAN", {"default": False}),
                "detect_full_body": ("BOOLEAN", {"default": False}),
                "detect_person_segmentation": ("BOOLEAN", {"default": False}),
                "confidence": ("FLOAT", {"default": 0.30, "min": 0.1, "max": 1.0, "step": 0.05}),
                "sort_by": (["Confidence", "Area: Largest to Smallest", "Position: Left to Right", "Position: Top to Bottom", "Position: Nearest to Center"],),
                "max_objects": ("INT", {"default": 10, "min": 1, "max": 10, "step": 1}),
                "mask_padding": ("INT", {"default": 35, "min": 0, "max": 200, "step": 1}),
                "mask_blur": ("INT", {"default": 6, "min": 0, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("SEGS", "IMAGE")
    RETURN_NAMES = ("SEGS", "IMAGE_PASSTHROUGH")
    FUNCTION = "process"
    CATEGORY = "HandFixerSuite"

    def process(self, image, detect_hands, detect_face, detect_eyes, detect_mouth, detect_feet, detect_torso, detect_full_body, detect_person_segmentation, confidence, sort_by, max_objects, mask_padding, mask_blur):
        global ENGINE_INSTANCE
        if ENGINE_INSTANCE is None:
            ENGINE_INSTANCE = Mediapipe_Engine()

        np_image = (image[0].numpy() * 255).astype(np.uint8)
        targets = []
        if detect_hands: targets.append("hands")
        if detect_face: targets.append("face")
        if detect_eyes: targets.append("eyes")
        if detect_mouth: targets.append("mouth")
        if detect_feet: targets.append("feet")
        if detect_torso: targets.append("torso")
        if detect_full_body: targets.append("full_body (bbox)")
        if detect_person_segmentation: targets.append("person (segmentation)")

        segs = []
        for model_type in targets:
            options = {
                'model_type': model_type,
                'sort_by': sort_by,
                'max_objects': max_objects,
                'confidence': confidence,
                'mask_padding': mask_padding,
                'mask_blur': mask_blur
            }
            mask_np = ENGINE_INSTANCE.process_and_create_mask(np_image, options)
            mask_tensor = torch.from_numpy(mask_np.astype(np.float32) / 255.0).unsqueeze(0)
            segs.append(mask_tensor)

        return (segs, image)

NODE_CLASS_MAPPINGS = {
    "Luna_MediaPipe_Segs": Luna_MediaPipe_Segs
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Luna_MediaPipe_Segs": "Luna MediaPipe SEGS"
}
