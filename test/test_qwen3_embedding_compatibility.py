"""
Test script to verify embedding compatibility between:
- Qwen3-4B (Z-IMAGE's CLIP encoder)
- Qwen3-VL-4B (Vision-Language model)

If embeddings are compatible, we can use ONE model for both:
1. Text encoding for Z-IMAGE
2. Vision/captioning for image understanding

Run from ComfyUI venv:
    python test/test_qwen3_embedding_compatibility.py
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Paths to your models
QWEN3_4B_SAFETENSORS = r"D:\AI\SD Models\text_encoders\qwen_3_4b.safetensors"
QWEN3_VL_4B_DIR = r"F:\LLM\Models\huihui\Qwen3-VL-4B-instruct-abliterated"
QWEN3_4B_GGUF = r"C:\Users\bcemm\.lmstudio\Models\Lockout\qwen3-4b-heretic-zimage\qwen-4b-zimage-heretic-q8.gguf"


def load_safetensors_embeddings(path: str) -> dict:
    """Load embedding weights from safetensors file"""
    try:
        from safetensors import safe_open
        
        embeddings = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = f.keys()
            print(f"\n📁 Safetensors keys ({len(keys)} total):")
            
            # Find embedding-related keys
            embed_keys = [k for k in keys if 'embed' in k.lower() or 'wte' in k.lower()]
            for k in embed_keys[:10]:  # First 10
                print(f"   {k}: {f.get_tensor(k).shape}")
                embeddings[k] = f.get_tensor(k)
                
            # Also check for any lm_head or output embeddings
            other_keys = [k for k in keys if 'lm_head' in k.lower() or 'output' in k.lower()][:5]
            for k in other_keys:
                print(f"   {k}: {f.get_tensor(k).shape}")
                
        return embeddings
    except Exception as e:
        print(f"❌ Error loading safetensors: {e}")
        return {}


def load_gguf_info(path: str):
    """Get info from GGUF file (requires gguf package)"""
    try:
        # Try to use llama-cpp-python or gguf package
        print(f"\n📁 GGUF file: {path}")
        print(f"   Size: {os.path.getsize(path) / 1e9:.2f} GB")
        
        # Try gguf package
        try:
            import gguf
            reader = gguf.GGUFReader(path)
            print(f"   Architecture: {reader.fields.get('general.architecture', 'unknown')}")
            print(f"   Vocab size: {reader.fields.get('tokenizer.ggml.vocab_size', 'unknown')}")
        except (ImportError, Exception) as pkg_err:
            print(f"   (Install 'gguf' package for detailed info: {pkg_err})")
            
    except Exception as e:
        print(f"❌ Error reading GGUF: {e}")


def check_transformers_model(model_dir: str):
    """Check a HuggingFace transformers model directory"""
    try:
        from transformers import AutoConfig, AutoTokenizer
        
        print(f"\n📁 Checking HF model: {model_dir}")
        
        # Load config
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        print(f"   Model type: {config.model_type}")
        print(f"   Vocab size: {config.vocab_size}")
        print(f"   Hidden size: {config.hidden_size}")
        
        # Check for vision config
        if hasattr(config, 'vision_config'):
            print(f"   Has vision config: Yes")
            print(f"   Vision hidden size: {config.vision_config.hidden_size}")
        else:
            print(f"   Has vision config: No (text-only)")
            
        # Load tokenizer to compare
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        print(f"   Tokenizer vocab size: {tokenizer.vocab_size}")
        
        return config, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model config: {e}")
        return None, None


def compare_tokenizers(tok1, tok2, test_texts: list[str]):
    """Compare tokenization between two tokenizers"""
    print("\n🔍 Comparing tokenization:")
    
    for text in test_texts:
        ids1 = tok1.encode(text)
        ids2 = tok2.encode(text)
        
        match = ids1 == ids2
        print(f"   '{text[:50]}...' → {'✅ MATCH' if match else '❌ DIFFER'}")
        if not match:
            print(f"      tok1: {ids1[:10]}...")
            print(f"      tok2: {ids2[:10]}...")


def compare_embedding_weights(emb1: torch.Tensor, emb2: torch.Tensor) -> dict:
    """Compare two embedding weight matrices"""
    results = {}
    
    # Shape comparison
    results['shape_match'] = emb1.shape == emb2.shape
    results['shape1'] = emb1.shape
    results['shape2'] = emb2.shape
    
    if not results['shape_match']:
        print(f"   ⚠️ Shape mismatch: {emb1.shape} vs {emb2.shape}")
        return results
    
    # Numerical comparison
    diff = (emb1 - emb2).abs()
    results['mean_diff'] = diff.mean().item()
    results['max_diff'] = diff.max().item()
    results['std_diff'] = diff.std().item()
    
    # Cosine similarity of random samples
    num_samples = min(1000, emb1.shape[0])
    indices = torch.randperm(emb1.shape[0])[:num_samples]
    
    cos_sims = []
    for idx in indices:
        v1 = emb1[idx].float()
        v2 = emb2[idx].float()
        cos_sim = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
        cos_sims.append(cos_sim.item())
    
    results['mean_cosine_sim'] = np.mean(cos_sims)
    results['min_cosine_sim'] = np.min(cos_sims)
    
    return results


def main():
    print("=" * 70)
    print("🧪 Qwen3 Embedding Compatibility Test")
    print("=" * 70)
    
    # Test 1: Check safetensors file
    print("\n" + "=" * 70)
    print("📊 Test 1: Analyze Z-IMAGE CLIP encoder (qwen_3_4b.safetensors)")
    print("=" * 70)
    
    if os.path.exists(QWEN3_4B_SAFETENSORS):
        embeddings_clip = load_safetensors_embeddings(QWEN3_4B_SAFETENSORS)
    else:
        print(f"❌ File not found: {QWEN3_4B_SAFETENSORS}")
        embeddings_clip = {}
    
    # Test 2: Check Qwen3-VL-4B
    print("\n" + "=" * 70)
    print("📊 Test 2: Analyze Qwen3-VL-4B model")
    print("=" * 70)
    
    config_vl, tokenizer_vl = None, None
    if os.path.exists(QWEN3_VL_4B_DIR):
        config_vl, tokenizer_vl = check_transformers_model(QWEN3_VL_4B_DIR)
    else:
        print(f"❌ Directory not found: {QWEN3_VL_4B_DIR}")
    
    # Test 3: Check GGUF file
    print("\n" + "=" * 70)
    print("📊 Test 3: Analyze Qwen3-4B GGUF (Z-IMAGE heretic)")
    print("=" * 70)
    
    if os.path.exists(QWEN3_4B_GGUF):
        load_gguf_info(QWEN3_4B_GGUF)
    else:
        print(f"❌ File not found: {QWEN3_4B_GGUF}")
    
    # Test 4: Try to load and compare embeddings directly
    print("\n" + "=" * 70)
    print("📊 Test 4: Direct embedding weight comparison")
    print("=" * 70)
    
    try:
        from transformers import AutoModel
        
        if os.path.exists(QWEN3_VL_4B_DIR):
            print("\n⏳ Loading Qwen3-VL-4B model (this may take a moment)...")
            model_vl = AutoModel.from_pretrained(
                QWEN3_VL_4B_DIR, 
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="cpu"
            )
            
            # Get embedding layer
            if hasattr(model_vl, 'model') and hasattr(model_vl.model, 'embed_tokens'):
                emb_vl = model_vl.model.embed_tokens.weight.data.cpu()
                print(f"   VL embedding shape: {emb_vl.shape}")
            elif hasattr(model_vl, 'get_input_embeddings'):
                emb_vl = model_vl.get_input_embeddings().weight.data.cpu()
                print(f"   VL embedding shape: {emb_vl.shape}")
            else:
                print("   ⚠️ Could not find embedding layer in VL model")
                emb_vl = None
            
            # Compare with safetensors embeddings
            if embeddings_clip and emb_vl is not None:
                # Find the embedding tensor in safetensors
                embed_key = None
                for k in embeddings_clip:
                    if 'embed_tokens' in k or 'wte' in k:
                        embed_key = k
                        break
                
                if embed_key:
                    emb_clip = embeddings_clip[embed_key].cpu()
                    print(f"\n📊 Comparing embeddings:")
                    print(f"   CLIP key: {embed_key}")
                    print(f"   CLIP shape: {emb_clip.shape}")
                    print(f"   VL shape: {emb_vl.shape}")
                    
                    results = compare_embedding_weights(emb_clip.float(), emb_vl.float())
                    
                    print(f"\n📈 Results:")
                    print(f"   Shape match: {results['shape_match']}")
                    print(f"   Mean absolute diff: {results.get('mean_diff', 'N/A'):.6f}")
                    print(f"   Max absolute diff: {results.get('max_diff', 'N/A'):.6f}")
                    print(f"   Mean cosine similarity: {results.get('mean_cosine_sim', 'N/A'):.6f}")
                    print(f"   Min cosine similarity: {results.get('min_cosine_sim', 'N/A'):.6f}")
                    
                    # Verdict
                    print("\n" + "=" * 70)
                    print("🎯 VERDICT")
                    print("=" * 70)
                    
                    if results.get('mean_cosine_sim', 0) > 0.99:
                        print("✅ Embeddings are HIGHLY COMPATIBLE (>99% cosine similarity)")
                        print("   → Safe to use Qwen3-VL-4B for both CLIP encoding and vision!")
                    elif results.get('mean_cosine_sim', 0) > 0.95:
                        print("⚠️ Embeddings are MOSTLY COMPATIBLE (>95% cosine similarity)")
                        print("   → May work but test image generation quality")
                    elif results.get('mean_diff', float('inf')) < 0.01:
                        print("✅ Embeddings are VERY SIMILAR (mean diff < 0.01)")
                        print("   → Safe to use Qwen3-VL-4B for both CLIP encoding and vision!")
                    else:
                        print("❌ Embeddings DIFFER significantly")
                        print("   → Need separate models for CLIP and vision")
                        
            del model_vl
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✨ Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
