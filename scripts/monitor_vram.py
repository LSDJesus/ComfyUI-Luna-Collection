"""
Luna VRAM Monitor - Track GPU memory usage during workflows
Usage: python scripts/monitor_vram.py [interval_seconds] [output_file]
Default: 0.5 second interval, saves to vram_monitor.csv
"""

import time
import csv
import sys
from datetime import datetime
import os

# Try pynvml first (more reliable for monitoring), fallback to torch
try:
    import pynvml
    USE_PYNVML = True
except ImportError:
    import torch
    USE_PYNVML = False
    print("Warning: pynvml not available, using torch (less reliable)")
    print("Install with: pip install nvidia-ml-py\n")

def get_gpu_memory():
    """Get memory info for all GPUs using native NVIDIA API"""
    if USE_PYNVML:
        return get_gpu_memory_pynvml()
    else:
        return get_gpu_memory_torch()

def get_gpu_memory_pynvml():
    """Get memory using pynvml (NVIDIA Management Library)"""
    gpu_info = []
    
    try:
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        
        for gpu_id in range(gpu_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                used_mem = mem_info.used
                total_mem = mem_info.total
                free_mem = mem_info.free
                
                used_gb = used_mem / 1024**3
                total_gb = total_mem / 1024**3
                percent = (used_mem / total_mem) * 100
                
                gpu_info.append({
                    'gpu_id': gpu_id,
                    'used_gb': used_gb,
                    'total_gb': total_gb,
                    'percent': percent,
                    'used_mb': used_mem / 1024**2,
                    'free_mb': free_mem / 1024**2
                })
            except Exception as e:
                gpu_info.append({
                    'gpu_id': gpu_id,
                    'error': str(e)
                })
    except Exception as e:
        print(f"Error initializing NVML: {e}")
    
    return gpu_info

def get_gpu_memory_torch():
    """Fallback: Get memory using torch (less reliable for monitoring)"""
    if not torch.cuda.is_available():
        return []
    
    gpu_count = torch.cuda.device_count()
    gpu_info = []
    
    for gpu_id in range(gpu_count):
        try:
            # Force synchronization to get fresh memory stats
            torch.cuda.synchronize(gpu_id)
            
            free_mem, total_mem = torch.cuda.mem_get_info(gpu_id)
            used_mem = total_mem - free_mem
            used_gb = used_mem / 1024**3
            total_gb = total_mem / 1024**3
            percent = (used_mem / total_mem) * 100
            
            gpu_info.append({
                'gpu_id': gpu_id,
                'used_gb': used_gb,
                'total_gb': total_gb,
                'percent': percent,
                'used_mb': used_mem / 1024**2,
                'free_mb': free_mem / 1024**2
            })
        except Exception as e:
            gpu_info.append({
                'gpu_id': gpu_id,
                'error': str(e)
            })
    
    return gpu_info

def format_terminal_output(timestamp, gpu_info):
    """Format colored terminal output"""
    parts = [f"[{timestamp}]"]
    
    for gpu in gpu_info:
        if 'error' in gpu:
            parts.append(f"GPU{gpu['gpu_id']}: ERROR")
            continue
        
        # Color based on usage
        percent = gpu['percent']
        if percent > 90:
            color = '\033[91m'  # Red
        elif percent > 70:
            color = '\033[93m'  # Yellow
        else:
            color = '\033[92m'  # Green
        reset = '\033[0m'
        
        parts.append(f"{color}GPU{gpu['gpu_id']}: {gpu['used_gb']:.2f}/{gpu['total_gb']:.1f}GB ({percent:.1f}%){reset}")
    
    return " | ".join(parts)

def monitor_vram(interval=0.5, output_file="vram_monitor.csv"):
    """Monitor VRAM usage and log to CSV"""
    print(f"Luna VRAM Monitor")
    api_name = "pynvml (native NVIDIA API)" if USE_PYNVML else "torch (may not show real-time changes)"
    print(f"Using: {api_name}")
    print(f"Interval: {interval}s")
    print(f"Output: {output_file}")
    print(f"Press Ctrl+C to stop\n")
    
    # Get GPU count for headers
    if USE_PYNVML:
        try:
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
        except:
            gpu_count = 0
    else:
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # Create output file with headers
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ['timestamp', 'elapsed_sec']
        for gpu_id in range(gpu_count):
            headers.extend([
                f'gpu{gpu_id}_used_gb',
                f'gpu{gpu_id}_total_gb',
                f'gpu{gpu_id}_percent',
                f'gpu{gpu_id}_used_mb',
                f'gpu{gpu_id}_free_mb'
            ])
        writer.writerow(headers)
    
    start_time = time.time()
    record_count = 0
    
    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            gpu_info = get_gpu_memory()
            
            # Write to CSV (unbuffered)
            with open(output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                row = [timestamp, f"{elapsed:.3f}"]
                
                for gpu in gpu_info:
                    if 'error' in gpu:
                        row.extend(['ERROR', 'ERROR', 'ERROR', 'ERROR', 'ERROR'])
                    else:
                        row.extend([
                            f"{gpu['used_gb']:.3f}",
                            f"{gpu['total_gb']:.3f}",
                            f"{gpu['percent']:.2f}",
                            f"{gpu['used_mb']:.2f}",
                            f"{gpu['free_mb']:.2f}"
                        ])
                
                writer.writerow(row)
            
            # Print to terminal
            print(format_terminal_output(timestamp, gpu_info))
            
            record_count += 1
            
            # Sleep until next interval
            time.sleep(max(0, interval - (time.time() - current_time)))
            
    except KeyboardInterrupt:
        print(f"\n\nMonitoring stopped.")
        print(f"Records captured: {record_count}")
        print(f"Duration: {time.time() - start_time:.2f}s")
        print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    interval = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    output_file = sys.argv[2] if len(sys.argv) > 2 else "vram_monitor.csv"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    
    monitor_vram(interval, output_file)
