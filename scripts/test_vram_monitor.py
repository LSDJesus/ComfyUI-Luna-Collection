"""
Test VRAM monitoring for Luna Daemon Panel

This script tests:
1. Daemon VRAM tracking via torch.cuda.mem_get_info
2. ComfyUI model_management VRAM tracking
3. Weight registry model details
"""

import sys
import os
from pathlib import Path

# Add luna_daemon to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

print("=" * 60)
print("Luna VRAM Monitor Test")
print("=" * 60)

# Test 1: PyTorch VRAM tracking (what daemon uses)
print("\n[1] Testing PyTorch CUDA VRAM tracking...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            used = total - free
            print(f"  GPU {i}:")
            print(f"    Total: {total / 1024**3:.2f} GB")
            print(f"    Used:  {used / 1024**3:.2f} GB")
            print(f"    Free:  {free / 1024**3:.2f} GB")
            print(f"    Usage: {(used / total * 100):.1f}%")
    else:
        print("✗ CUDA not available")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Daemon client connection
print("\n[2] Testing Daemon connection...")
try:
    from luna_daemon import client as daemon_client
    
    if daemon_client.is_daemon_running():
        print("✓ Daemon is running")
        
        info = daemon_client.get_daemon_info()
        print(f"  Devices: {info.get('devices', {})}")
        print(f"  VRAM tracking: {len(info.get('vram', {}))} GPU(s)")
        
        for device_id, vram in info.get('vram', {}).items():
            print(f"  {device_id}:")
            print(f"    Used:  {vram.get('used_gb', 0):.2f} GB")
            print(f"    Total: {vram.get('total_gb', 0):.2f} GB")
        
        # Check weight registry
        wr_models = info.get('weight_registry_models', [])
        if wr_models:
            print(f"\n  Weight Registry: {len(wr_models)} model(s) loaded")
            for model in wr_models:
                print(f"    {model['type']}: {model['memory_gb']:.3f} GB ({model['tensor_count']} tensors)")
        else:
            print(f"  Weight Registry: No models loaded yet")
    else:
        print("✗ Daemon not running")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: ComfyUI model_management (if available)
print("\n[3] Testing ComfyUI model_management VRAM tracking...")
try:
    import comfy.model_management as mm
    
    device = mm.get_torch_device()
    print(f"✓ ComfyUI device: {device}")
    
    if hasattr(mm, 'get_total_memory'):
        total = mm.get_total_memory(device) / 1024**3
        free = mm.get_free_memory(device) / 1024**3
        used = total - free
        print(f"  Total: {total:.2f} GB")
        print(f"  Used:  {used:.2f} GB")
        print(f"  Free:  {free:.2f} GB")
        print(f"  Usage: {(used / total * 100):.1f}%")
    else:
        print("  ⚠ get_total_memory not available")
    
    if hasattr(mm, 'current_loaded_models'):
        models = mm.current_loaded_models
        print(f"  Loaded models: {len(models)}")
    else:
        print("  ⚠ current_loaded_models not available")
        
except ImportError:
    print("✗ ComfyUI not in path (expected if run outside ComfyUI)")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
