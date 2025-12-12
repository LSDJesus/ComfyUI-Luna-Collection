import torch
import math

def luna_tiling_orchestrator(image_tensor, model, tile_x, tile_y, overlap, strategy):
    with torch.inference_mode():  # Disable gradient tracking
        height, width = image_tensor.shape[2], image_tensor.shape[3]
        scale = model.scale
        new_height, new_width = int(height * scale), int(width * scale)
        
        device = image_tensor.device
        
        # Estimate output size in GB
        output_bytes = image_tensor.shape[0] * image_tensor.shape[1] * new_height * new_width * 4
        output_gb = output_bytes / (1024**3)
        
        # Use CPU canvas only for truly massive outputs (>8GB) - allows 4K→16K on high-VRAM GPUs
        canvas_device = 'cpu' if output_gb > 8.0 else device
        canvas = torch.zeros((image_tensor.shape[0], image_tensor.shape[1], new_height, new_width), 
                           device=canvas_device, dtype=image_tensor.dtype)
    
    rows = math.ceil(height / tile_y)
    cols = math.ceil(width / tile_x)
    
    pass_one_tiles = []
    pass_two_tiles = []

    for r in range(rows):
        for c in range(cols):
            tile = (c * tile_x, r * tile_y, tile_x, tile_y)
            if strategy == 'chess' and (r + c) % 2 == 1:
                pass_two_tiles.append(tile)
            else:
                pass_one_tiles.append(tile)
    
    for tile_list in [pass_one_tiles, pass_two_tiles]:
        if not tile_list: continue
        
        for x, y, tile_w, tile_h in tile_list:
            y_start = max(0, y - overlap); x_start = max(0, x - overlap)
            y_end = min(height, y + tile_h + overlap); x_end = min(width, x + tile_w + overlap)

            tile_input = image_tensor[:, :, y_start:y_end, x_start:x_end]
            upscaled_tile = model(tile_input)

            paste_y = y * scale; paste_x = x * scale
            
            crop_y_start = (y - y_start) * scale; crop_x_start = (x - x_start) * scale
            
            paste_h = min(new_height - paste_y, tile_h * scale)
            paste_w = min(new_width - paste_x, tile_w * scale)
            
            crop_y_end = crop_y_start + paste_h
            crop_x_end = crop_x_start + paste_w
            
            # Extract crop and move to canvas device (CPU or GPU)
            tile_crop = upscaled_tile[:, :, crop_y_start:crop_y_end, crop_x_start:crop_x_end].to(canvas_device)
            canvas[:, :, paste_y:paste_y+paste_h, paste_x:paste_x+paste_w] = tile_crop
            
            # Clean up GPU memory after each tile
            del upscaled_tile, tile_input, tile_crop
            if device != 'cpu':
                torch.cuda.empty_cache()
    
    # Move final canvas back to original device if needed
    if canvas_device != device:
        canvas = canvas.to(device)
    
    return canvas
