import torch
import numpy as np
import cv2
import os

class Luna_YOLO_Annotation_Exporter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "class_id": ("INT", {"default": 0, "min": 0, "step": 1}),
                "filename_prefix": ("STRING", {"default": "image_00001"}),
                "output_path": ("STRING", {"default": "D:\\AI\\ComfyUI\\output\\yolo_annotations"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "export_annotation"
    OUTPUT_NODE = True
    CATEGORY = "Luna Collection/IO"

    def export_annotation(self, mask, class_id, filename_prefix, output_path):
        # --- 1. PREPARE THE MASK ---
        # Squeeze the batch dimension and move tensor to CPU
        mask_tensor = mask.cpu().squeeze(0)

        # Threshold the mask to create a binary image (0 or 1)
        # Anything > 0.5 is considered part of the mask
        binary_mask = (mask_tensor > 0.5).float()

        # Convert to a NumPy array in the format OpenCV expects (0-255, uint8)
        # Shape: [H, W]
        mask_np = (binary_mask.numpy() * 255).astype(np.uint8)

        # --- 2. FIND AND PROCESS CONTOURS ---
        # Find the contours of the segmented object
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print(f"[Luna Exporter] Warning: No contours found for {filename_prefix}. Skipping.")
            return {}

        # Assuming the largest contour is our target object
        largest_contour = max(contours, key=cv2.contourArea)

        # Get image dimensions for normalization
        height, width = mask_np.shape

        # --- 3. NORMALIZE AND FORMAT FOR YOLO ---
        # Flatten the contour array and normalize coordinates
        normalized_points = largest_contour.flatten().astype(float)
        normalized_points[0::2] /= width  # Normalize x coordinates
        normalized_points[1::2] /= height # Normalize y coordinates

        # Format the final string: class_id x1 y1 x2 y2 ...
        yolo_string = f"{class_id} " + " ".join(map(str, normalized_points))

        # --- 4. WRITE TO FILE ---
        # Ensure the output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Define the full output file path
        output_file = os.path.join(output_path, f"{filename_prefix}.txt")

        # Write the annotation
        with open(output_file, 'w') as f:
            f.write(yolo_string)

        print(f"[Luna Exporter] Successfully wrote annotation to {output_file}")

        return {}

# This dictionary is what ComfyUI uses to register the node
NODE_CLASS_MAPPINGS = {
    "Luna_YOLO_Annotation_Exporter": Luna_YOLO_Annotation_Exporter
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Luna_YOLO_Annotation_Exporter": "YOLO Annotation Exporter (Luna)"
}