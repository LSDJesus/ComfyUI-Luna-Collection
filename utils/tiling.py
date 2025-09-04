import torch
import math

def luna_tiling_orchestrator(image_tensor, model, tile_x, tile_y, overlap, strategy):
    height, width = image_tensor.shape[2], image_tensor.shape[3]
    scale = model.scale
    new_height, new_width = int(height * scale), int(width * scale)
    
    device = image_tensor.device
    canvas = torch.zeros((image_tensor.shape[0], image_tensor.shape[1], new_height, new_width), device=device)
    
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
            
            canvas[:, :, paste_y:paste_y+paste_h, paste_x:paste_x+paste_w] = upscaled_tile[:, :, crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    
    return canvas
