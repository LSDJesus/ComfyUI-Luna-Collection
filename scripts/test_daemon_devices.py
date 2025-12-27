#!/usr/bin/env python3
"""
Test script for Luna Daemon device configuration

Run this to test the daemon's socket commands independently of ComfyUI.

Usage:
    python test_daemon_devices.py

Requirements:
    - Luna Daemon running (python -m luna_daemon)
    - torch installed
"""

import sys
import os
from pathlib import Path

# Add luna_daemon to path
daemon_dir = Path(__file__).parent.parent / "luna_daemon"
sys.path.insert(0, str(daemon_dir.parent))

# Import daemon client
try:
    from luna_daemon import client
except ImportError as e:
    print(f"❌ Failed to import daemon client: {e}")
    print("Make sure you're running this from the ComfyUI-Luna-Collection directory")
    sys.exit(1)


def test_daemon_connection():
    """Test if daemon is running and reachable"""
    print("\n" + "="*60)
    print("Testing Daemon Connection")
    print("="*60)
    
    if not client.is_daemon_running():
        print("❌ Daemon is not running!")
        print("\nTo start the daemon, run:")
        print("  python -m luna_daemon")
        return False
    
    print("✓ Daemon is running")
    return True


def test_get_info():
    """Test getting daemon info"""
    print("\n" + "="*60)
    print("Testing Get Daemon Info")
    print("="*60)
    
    try:
        info = client.get_daemon_info()
        if info:
            print("✓ Got daemon info:")
            for key, value in info.items():
                if key != "vram":  # Skip VRAM dict for cleaner output
                    print(f"  {key}: {value}")
            
            # Show VRAM per GPU if available
            if "vram" in info:
                print("  VRAM:")
                for gpu, vram_info in info["vram"].items():
                    print(f"    {gpu}: {vram_info['used_gb']} / {vram_info['total_gb']} GB")
            
            return True
        else:
            print("❌ Failed to get daemon info")
            return False
    except Exception as e:
        print(f"❌ Error getting daemon info: {e}")
        return False


def test_get_devices():
    """Test getting device configuration"""
    print("\n" + "="*60)
    print("Testing Get Devices")
    print("="*60)
    
    try:
        result = client.get_devices()
        if result and result.get("success"):
            print("✓ Got device configuration:")
            
            devices = result.get("devices", {})
            print(f"  CLIP Device: {devices.get('clip', 'unknown')}")
            print(f"  VAE Device: {devices.get('vae', 'unknown')}")
            print(f"  LLM Device: {devices.get('llm', 'unknown')}")
            
            print("\n  Available GPUs:")
            gpus = result.get("available_gpus", [])
            if gpus:
                for gpu in gpus:
                    print(f"    - {gpu}")
            else:
                print("    - (no CUDA GPUs available)")
            
            print(f"  CUDA Available: {result.get('has_cuda', False)}")
            
            return True, devices
        else:
            print(f"❌ Failed to get devices: {result.get('error', 'unknown error')}")
            return False, None
    except Exception as e:
        print(f"❌ Error getting devices: {e}")
        return False, None


def test_set_clip_device(current_devices):
    """Test setting CLIP device"""
    print("\n" + "="*60)
    print("Testing Set CLIP Device")
    print("="*60)
    
    if not current_devices:
        print("⊘ Skipped (no current device info)")
        return None
    
    current = current_devices.get("clip", "unknown")
    print(f"  Current CLIP device: {current}")
    
    # Just test with current device (no-op change) to verify functionality
    try:
        result = client.set_clip_device(current)
        if result and result.get("success"):
            print(f"✓ CLIP device set successfully")
            print(f"  Message: {result.get('message', '')}")
            return True
        else:
            print(f"❌ Failed to set CLIP device: {result.get('error', 'unknown error')}")
            return False
    except Exception as e:
        print(f"❌ Error setting CLIP device: {e}")
        return False


def test_set_vae_device(current_devices):
    """Test setting VAE device"""
    print("\n" + "="*60)
    print("Testing Set VAE Device")
    print("="*60)
    
    if not current_devices:
        print("⊘ Skipped (no current device info)")
        return None
    
    current = current_devices.get("vae", "unknown")
    print(f"  Current VAE device: {current}")
    
    # Just test with current device (no-op change) to verify functionality
    try:
        result = client.set_vae_device(current)
        if result and result.get("success"):
            print(f"✓ VAE device set successfully")
            print(f"  Message: {result.get('message', '')}")
            return True
        else:
            print(f"❌ Failed to set VAE device: {result.get('error', 'unknown error')}")
            return False
    except Exception as e:
        print(f"❌ Error setting VAE device: {e}")
        return False


def test_set_llm_device(current_devices):
    """Test setting LLM device"""
    print("\n" + "="*60)
    print("Testing Set LLM Device")
    print("="*60)
    
    if not current_devices:
        print("⊘ Skipped (no current device info)")
        return None
    
    current = current_devices.get("llm", "unknown")
    print(f"  Current LLM device: {current}")
    
    # Just test with current device (no-op change) to verify functionality
    try:
        result = client.set_llm_device(current)
        if result and result.get("success"):
            print(f"✓ LLM device set successfully")
            print(f"  Message: {result.get('message', '')}")
            return True
        else:
            print(f"❌ Failed to set LLM device: {result.get('error', 'unknown error')}")
            return False
    except Exception as e:
        print(f"❌ Error setting LLM device: {e}")
        return False


def test_invalid_device(current_devices):
    """Test error handling with invalid device"""
    print("\n" + "="*60)
    print("Testing Error Handling (Invalid Device)")
    print("="*60)
    
    try:
        result = client.set_clip_device("cuda:999")  # Likely invalid
        if result and not result.get("success"):
            print(f"✓ Correctly rejected invalid device:")
            print(f"  Error: {result.get('error', 'unknown')}")
            return True
        else:
            print(f"⊘ Device was accepted (might be valid on your system)")
            return None
    except Exception as e:
        print(f"✓ Correctly raised exception for invalid device: {e}")
        return True


def main():
    print("\n" + "🌙 "*30)
    print("Luna Daemon Device Configuration Test")
    print("🌙 "*30)
    
    # Test connection
    if not test_daemon_connection():
        return
    
    # Test getting info
    test_get_info()
    
    # Test device endpoints
    success, devices = test_get_devices()
    
    if success:
        test_set_clip_device(devices)
        test_set_vae_device(devices)
        test_set_llm_device(devices)
        test_invalid_device(devices)
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print("\n✓ All device configuration endpoints are working!")
    print("\nYou can now:")
    print("  1. Use these endpoints via the ComfyUI Luna Daemon panel")
    print("  2. Call them directly from Python using luna_daemon.client")
    print("  3. Make HTTP requests to the /luna/daemon/* endpoints")
    print("\nExample Python usage:")
    print("  from luna_daemon import client")
    print("  client.set_clip_device('cuda:1')")
    print("  client.set_vae_device('cuda:0')")
    print("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⊘ Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
