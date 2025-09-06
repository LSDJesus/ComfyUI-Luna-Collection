import torch
import numpy as np
from PIL import Image
import os
from nodes.luna_mediapipe_segs import Luna_MediaPipe_Segs

def load_image(path):
    img = Image.open(path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # Shape: (1, H, W, C)
    return img_tensor, img_np

if __name__ == "__main__":
    img_path = "test/test2.png"
    img_tensor, img_np = load_image(img_path)
    node = Luna_MediaPipe_Segs()
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
        confidence=0.25,
        sort_by="Confidence",
        max_objects=5,
        mask_padding=25,
        mask_blur=6
    )
    os.makedirs("test/output", exist_ok=True)
    for i, mask in enumerate(segs):
        mask_np = mask.squeeze().numpy()
        # Apply mask to image
        masked_img = img_np * mask_np[..., None]
        masked_img = (masked_img * 255).astype(np.uint8)
        out_img = Image.fromarray(masked_img)
        out_img.save(f"test/output/test_masked_object_{i}.png")
    print(f"Mask {i} max value: {mask_np.max()}, min value: {mask_np.min()}")
    print(f"Saved {len(segs)} masked images to test/output/")
