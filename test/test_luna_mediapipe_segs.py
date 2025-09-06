import torch
import numpy as np
from PIL import Image
from nodes.luna_mediapipe_segs import Luna_MediaPipe_Segs

# Load and preprocess image
def load_image(path):
    img = Image.open(path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # Shape: (1, H, W, C)
    return img_tensor

if __name__ == "__main__":
    img_tensor = load_image("test.png")
    node = Luna_MediaPipe_Segs()
    segs, img_out = node.process(
        image=img_tensor,
        detect_hands=True,
        detect_face=True,
        detect_eyes=False,
        detect_mouth=False,
        detect_feet=False,
        detect_torso=False,
        detect_full_body=False,
        detect_person_segmentation=False,
        confidence=0.3,
        sort_by="Confidence",
        max_objects=5,
        mask_padding=35,
        mask_blur=6
    )
    print(f"Number of SEGS masks: {len(segs)}")
    for i, mask in enumerate(segs):
        print(f"Mask {i} shape: {mask.shape}")
