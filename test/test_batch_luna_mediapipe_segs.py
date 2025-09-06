import torch
import numpy as np
from PIL import Image
import os
from nodes.luna_mediapipe_segs import Luna_MediaPipe_Segs

def load_image(path):
    img = Image.open(path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # Shape: (1, H, W, C)
    return img_tensor

if __name__ == "__main__":
    test_dir = os.path.dirname(__file__)
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    node = Luna_MediaPipe_Segs()
    for img_file in image_files:
        print(f"Testing {img_file}...")
        img_tensor = load_image(os.path.join(test_dir, img_file))
        segs, img_out = node.process(
            image=img_tensor,
            detect_hands=True,
            detect_face=True,
            detect_eyes=True,
            detect_mouth=True,
            detect_feet=True,
            detect_torso=True,
            detect_full_body=True,
            detect_person_segmentation=True,
            confidence=0.3,
            sort_by="Confidence",
            max_objects=5,
            mask_padding=35,
            mask_blur=6
        )
        print(f"  Number of SEGS masks: {len(segs)}")
        for i, mask in enumerate(segs):
            print(f"    Mask {i} shape: {mask.shape}")
    print("Batch test complete.")
