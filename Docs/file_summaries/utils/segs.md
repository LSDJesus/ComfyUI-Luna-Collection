# utils/segs.py

## Purpose
Segmentation utilities providing mask manipulation, region cropping/pasting, and segment data structures for image processing workflows.

## Exports
- `SEG`: Named tuple for segment data (cropped_image, cropped_mask, confidence, crop_region, bbox, label, control_net_wrapper)
- `make_2d_mask(mask)`: Convert mask to 2D format regardless of input dimensions
- `crop_ndarray4(image, crop_region)`: Crop 4D ndarray using region coordinates
- `paste_ndarray4(dest, src, crop_region)`: Paste source array into destination at crop region
- `tensor_putalpha(tensor, mask)`: Add alpha channel to tensor using mask
- `segs_scale_match(segs, target_shape)`: Scale segment coordinates to match target dimensions
- `segs_to_combined_mask(segs)`: Combine all segment masks into single binary mask
- `segs_bitwise_and_mask(segs, mask)`: Apply bitwise AND operation between segments and mask
- `make_sam_mask(...)`: Placeholder for SAM (Segment Anything Model) integration

## Key Imports
- `collections`: namedtuple for SEG structure
- `torch`: Tensor operations
- `numpy`: Array manipulation

## ComfyUI Node Configuration
N/A - Utility functions, not a node.

## Input Schema
N/A - Utility functions.

## Key Methods
- `make_2d_mask(mask)`: Handle various mask dimensions (2D, 3D, 4D)
- `crop_ndarray4(image, crop_region)`: Extract rectangular region from 4D array
- `segs_scale_match(segs, target_shape)`: Resize segment coordinates proportionally
- `segs_to_combined_mask(segs)`: Merge multiple segment masks into single mask
- `segs_bitwise_and_mask(segs, mask)`: Filter segments using external mask

## Dependencies
- `torch`: Tensor operations for alpha channel handling
- `numpy`: Array operations for mask manipulation

## Integration Points
- Used by detailing and segmentation nodes
- Compatible with ComfyUI-Impact-Pack segment format
- Provides mask preprocessing for inpainting workflows
- Supports region-based image manipulation

## Notes
- SEG namedtuple matches ComfyUI-Impact-Pack format
- Handles various tensor dimensions automatically
- Includes placeholder for SAM integration (not implemented locally)
- Focuses on mask geometry operations and segment filtering
- Used by Luna detailing nodes for region-specific processing