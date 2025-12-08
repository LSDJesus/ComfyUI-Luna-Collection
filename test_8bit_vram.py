"""Test 8-bit quantized Qwen3-VL VRAM usage"""
import sys
sys.path.insert(0, '.')
from luna_daemon.qwen3_encoder import Qwen3VLEncoder, Qwen3VLConfig
import torch

print('Loading 8-bit quantized Qwen3-VL...')
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

config = Qwen3VLConfig(
    model_path=r'D:\AI\SD Models\LLM\Huihui-Qwen3-VL-4B-Instruct-abliterated',
    device='cuda:0',
    load_in_8bit=True
)

encoder = Qwen3VLEncoder(config)
encoder.load_model()

print('\nRunning inference...')
for i in range(5):
    embeddings = encoder.encode_text(f'test prompt number {i}')

peak_mem = torch.cuda.max_memory_allocated(0) / (1024**3)
print(f'\n✅ Peak VRAM usage: {peak_mem:.2f} GB')
print(f'Available on 3080Ti: 9.2 GB (12 GB - 2.8 GB desktop)')
print(f'Fits: {"✅ YES" if peak_mem < 9.2 else "❌ NO"}')
print()
print('Room for CLIP-L + CLIP-G: ~1.9 GB')
print(f'Total estimated: {peak_mem + 1.9:.2f} GB')
