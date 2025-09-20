from collections import namedtuple
import torch
import numpy as np

SEG = namedtuple("SEG",
                 ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                 defaults=[None])

def make_2d_mask(mask):
    if mask is None:
        return None
    if mask.dim() == 2:
        return mask
    elif mask.dim() == 3:
        return mask.squeeze(0)
    elif mask.dim() == 4:
        return mask.squeeze(0).squeeze(0)
    else:
        raise ValueError("Unsupported mask dimensions")

def crop_ndarray4(image, crop_region):
    x1, y1, x2, y2 = crop_region
    return image[y1:y2, x1:x2]

def paste_ndarray4(dest, src, crop_region):
    x1, y1, x2, y2 = crop_region
    dest[y1:y2, x1:x2] = src

def tensor_putalpha(tensor, mask):
    if mask.dim() == 3:
        mask = mask.squeeze(0)
    if tensor.shape[0] == 4:
        tensor[3:4] = mask
    else:
        tensor = torch.cat([tensor, mask.unsqueeze(0)], dim=0)
    return tensor

def segs_scale_match(segs, target_shape):
    h = segs[0][0]
    w = segs[0][1]

    th = target_shape[1]
    tw = target_shape[2]

    if (h == th and w == tw) or h == 0 or w == 0:
        return segs

    rh = th / h
    rw = tw / w

    new_segs = []
    for seg in segs[1]:
        new_crop_region = [int(seg.crop_region[0] * rw), int(seg.crop_region[1] * rh), int(seg.crop_region[2] * rw), int(seg.crop_region[3] * rh)]
        new_bbox = [int(seg.bbox[0] * rw), int(seg.bbox[1] * rh), int(seg.bbox[2] * rw), int(seg.bbox[3] * rh)]
        new_seg = SEG(
            cropped_image=seg.cropped_image,
            cropped_mask=seg.cropped_mask,
            confidence=seg.confidence,
            crop_region=new_crop_region,
            bbox=new_bbox,
            label=seg.label,
            control_net_wrapper=seg.control_net_wrapper
        )
        new_segs.append(new_seg)

    return (segs[0], new_segs)

def segs_to_combined_mask(segs):
    shape = segs[0]
    h = shape[0]
    w = shape[1]

    mask = np.zeros((h, w), dtype=np.uint8)

    for seg in segs[1]:
        mask[seg.cropped_mask > 0] = 255

    return torch.from_numpy(mask.astype(np.float32) / 255.0)

def segs_bitwise_and_mask(segs, mask):
    mask = make_2d_mask(mask)

    if mask is None:
        return segs

    items = []

    mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)

    for seg in segs[1]:
        cropped_mask = seg.cropped_mask
        if cropped_mask is not None:
            combined = np.bitwise_and(cropped_mask, mask_np)
            if combined.max() > 0:
                items.append(SEG(
                    cropped_image=seg.cropped_image,
                    cropped_mask=combined,
                    confidence=seg.confidence,
                    crop_region=seg.crop_region,
                    bbox=seg.bbox,
                    label=seg.label,
                    control_net_wrapper=seg.control_net_wrapper
                ))

    return (segs[0], items)

def make_sam_mask(sam, segs, image, detection_hint, dilation,
                  threshold, bbox_expansion, mask_hint_threshold, mask_hint_use_negative):
    # Simplified placeholder - in full implementation, this would need SAM logic
    # For now, return a dummy mask or raise NotImplementedError
    raise NotImplementedError("make_sam_mask not implemented in local utils. Use full impact pack.")

# Additional placeholders for other functions if needed
# Implement or import as necessary