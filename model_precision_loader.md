You have a massive task ahead of you, but automation is your friend here. Moving 358 checkpoints manually is madness; scripting the conversion is the only way.

Here are the two utilities you need.

### Utility 1: The "Blackwell Batcher" (FP16 $\to$ Native FP8)
**Target:** RTX 5090 (Main PC)
**Function:** Loads `.safetensors` checkpoints, identifies the UNet weights (`model.diffusion_model.*`), casts them to `float8_e4m3fn`, discards the rest (CLIP/VAE), and saves a new `.safetensors` file.

**Prerequisites:**
```powershell
uv pip install torch safetensors tqdm colorama
```

**Script:** `convert_fp8_unet.py`

```python
import os
import argparse
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from colorama import Fore, Style, init

init(autoreset=True)

def convert_to_fp8(input_path, output_path, dry_run=False):
    print(f"{Fore.CYAN}Processing: {os.path.basename(input_path)}")
    
    if dry_run:
        return

    try:
        # Load the checkpoint
        state_dict = load_file(input_path)
        new_dict = {}
        
        # Filter for UNet keys only (SDXL specific)
        # Keeps 'model.diffusion_model' prefix for compatibility with ComfyUI UNETLoader
        unet_keys = [k for k in state_dict.keys() if k.startswith("model.diffusion_model.")]
        
        if not unet_keys:
            print(f"{Fore.RED}Skipping: No UNet keys found (Is this an SDXL model?)")
            return

        print(f"  Found {len(unet_keys)} UNet layers. Converting to FP8_E4M3FN...")

        for key in unet_keys:
            tensor = state_dict[key]
            # Skip non-floating point tensors (like quantization stats or int64)
            if tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                # Cast to FP8 E4M3 (Best for Inference)
                new_dict[key] = tensor.to(torch.float8_e4m3fn)
            else:
                new_dict[key] = tensor

        # Save
        save_file(new_dict, output_path)
        
        # Calc compression
        orig_size = os.path.getsize(input_path) / (1024**3)
        new_size = os.path.getsize(output_path) / (1024**3)
        print(f"{Fore.GREEN}Success! {orig_size:.2f}GB -> {new_size:.2f}GB")

    except Exception as e:
        print(f"{Fore.RED}Error processing {input_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Batch Convert SDXL Checkpoints to FP8 UNet")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input directory containing .safetensors")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory for FP8 UNets")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually save files")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    files = [f for f in os.listdir(args.input) if f.endswith(".safetensors")]
    print(f"{Fore.YELLOW}Found {len(files)} checkpoints in {args.input}")

    for file in tqdm(files, desc="Batch Converting"):
        in_p = os.path.join(args.input, file)
        out_name = os.path.splitext(file)[0] + "_fp8_unet.safetensors"
        out_p = os.path.join(args.output, out_name)
        
        if os.path.exists(out_p):
            print(f"{Fore.YELLOW}Skipping {out_name} (Already exists)")
            continue
            
        convert_to_fp8(in_p, out_p, args.dry_run)

if __name__ == "__main__":
    main()
```

---

### Utility 2: The "Ampere Quantizer" (FP16 $\to$ GGUF UNet)
**Target:** RTX 3090 (Server PC)
**Function:** Wraps the standard `city96/ComfyUI-GGUF` conversion script to batch process your folder.

**Setup:**
Since you likely have `ComfyUI-GGUF` installed, we will leverage its internal conversion script.
1.  Locate: `D:\AI\ComfyUI\custom_nodes\ComfyUI-GGUF\tools\convert.py`
2.  (If you don't have it): `git clone https://github.com/city96/ComfyUI-GGUF` into a temp folder.

**Script:** `batch_gguf_convert.py`

```python
import os
import subprocess
import argparse
from tqdm import tqdm

# === CONFIGURATION ===
# Path to the city96 conversion script
CONVERTER_SCRIPT = r"D:\AI\ComfyUI\custom_nodes\ComfyUI-GGUF\tools\convert.py"
# =====================

def main():
    parser = argparse.ArgumentParser(description="Batch Convert SDXL to GGUF UNet")
    parser.add_argument("--input", "-i", required=True, help="Input Source Directory")
    parser.add_argument("--output", "-o", required=True, help="Output Destination Directory")
    parser.add_argument("--quant", "-q", default="Q8_0", choices=["Q8_0", "Q4_K_S", "Q5_K_M"], help="Quantization Type")
    args = parser.parse_args()

    if not os.path.exists(CONVERTER_SCRIPT):
        print(f"ERROR: Could not find converter script at: {CONVERTER_SCRIPT}")
        print("Please edit the script to point to your ComfyUI-GGUF/tools/convert.py")
        return

    os.makedirs(args.output, exist_ok=True)
    files = [f for f in os.listdir(args.input) if f.endswith(".safetensors")]

    print(f"Queueing {len(files)} models for {args.quant} conversion...")

    for file in tqdm(files):
        in_path = os.path.join(args.input, file)
        # Naming convention: modelname_Q8_0.gguf
        out_name = os.path.splitext(file)[0] + f"_{args.quant}.gguf"
        out_path = os.path.join(args.output, out_name)

        if os.path.exists(out_path):
            print(f"Skipping {out_name} (Exists)")
            continue

        # The Command
        # --unet-only automatically strips CLIP/VAE
        cmd = [
            "python", CONVERTER_SCRIPT,
            "--in", in_path,
            "--out", out_path,
            "--out-type", args.quant.lower(), # script expects lowercase (q8_0)
            "--unet-only" 
        ]

        try:
            # Run the conversion
            subprocess.run(cmd, check=True, shell=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to convert {file}: {e}")

if __name__ == "__main__":
    main()
```

### Execution Strategy

1.  **Mount the Drive:** Ensure your 8TB HDD is accessible (e.g., `Z:\Checkpoints`).
2.  **Run FP8 Conversion (For 5090):**
    ```powershell
    python convert_fp8_unet.py -i "Z:\Checkpoints" -o "D:\AI\ComfyUI\models\unets\SDXL_FP8"
    ```
3.  **Run GGUF Conversion (For 3090):**
    ```powershell
    python batch_gguf_convert.py -i "Z:\Checkpoints" -o "D:\AI\ComfyUI\models\unets\SDXL_GGUF" -q Q8_0
    ```

**Note regarding keys:**
The FP8 script preserves the `model.diffusion_model` prefix. In ComfyUI, you will load these using the **`UNETLoader`** node (or `LunaCheckpointLoader` if you updated it to handle standalone UNets), *not* the standard Checkpoint Loader. The GGUF files will load via the **`UnetLoaderGGUF`** node.


me: lol sorry i wasnt clear. im going to make custom nodes for this. it will have 4 inputs the first will be for the full precision model, the second for the output directory. ill symlink the `checkpoints` folder containing my library of fp16 models into the `models` location so they show up in comfyui automatically. then ill set the output directory to `models/checkpoints/converted` which will be a symlink back to a directory on my nvme. it will have a dropdown to select between bf16, fp8, gguf-Q8 and gguf-Q4_K_S. then it will have a strip vae/clip toggle that only becomes active if bf16/fp8 conversion is selected and not if gguf conversion is. then ill have a model output. the node will take the checkpoint, search for a match in the `models/checkpoints` root, if it doesnt find one it converts and then loads the converted model, if it finds one it just loads the already converted model. no node switching, no thinking and automatic loading of the re-precisioned model that is automatically moved to my nvme for quick loading


This is the ultimate "Lazy Engineer" solution (which is the best kind of engineering). You are essentially building a **JIT (Just-In-Time) Compiler for Checkpoints**.

Since you are adding this to the Luna Collection, I have written this using your standard structure. This node bridges the gap between `folder_paths`, `safetensors`, and the external `ComfyUI-GGUF` tools.

### The Node Design: `LunaDynamicModelLoader`

**Key Features:**
1.  **Smart Caching:** Checks `models/checkpoints/converted` first.
2.  **JIT Conversion:** If the converted model is missing, it creates it on the fly.
3.  **Format Agnostic:** Handles Safetensors (Native Load) and GGUF (via `ComfyUI-GGUF` logic).
4.  **Auto-Strip:** Automatically discards CLIP/VAE data for the optimized formats to save space.

**Requirements:**
*   You must have `ComfyUI-GGUF` installed (for the GGUF loading wrappers).
*   You need `gguf` installed in your venv (`uv pip install gguf`) for the conversion logic.

### `nodes/loaders/luna_dynamic_loader.py`

```python
import os
import torch
import folder_paths
import comfy.sd
import comfy.utils
from safetensors.torch import load_file, save_file
from tqdm import tqdm

# Attempt to import GGUF loader logic from installed custom_nodes
try:
    from custom_nodes.ComfyUI_GGUF.nodes import UnetLoaderGGUF
    HAS_GGUF_NODE = True
except ImportError:
    HAS_GGUF_NODE = False

class LunaDynamicModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "precision": (["bf16", "fp8_e4m3fn", "gguf_Q8_0", "gguf_Q4_K_S"],),
                "strip_components": ("BOOLEAN", {"default": True, "label_on": "Strip VAE/CLIP (UNet Only)", "label_off": "Keep Full Model"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "model_path")
    FUNCTION = "load_and_convert"
    CATEGORY = "Luna/Loaders"

    def load_and_convert(self, ckpt_name, precision, strip_components):
        # 1. Resolve Paths
        src_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        
        # Define the 'converted' directory (User must create/symlink this: models/checkpoints/converted)
        converted_root = os.path.join(folder_paths.models_dir, "checkpoints", "converted")
        os.makedirs(converted_root, exist_ok=True)

        # Construct Target Filename
        base_name = os.path.splitext(os.path.basename(ckpt_name))[0]
        
        # Determine Target Format details
        is_gguf = "gguf" in precision
        ext = ".gguf" if is_gguf else ".safetensors"
        
        # Suffix logic
        suffix = f"_{precision}"
        if strip_components and not is_gguf:
            suffix += "_unet"
        
        target_filename = f"{base_name}{suffix}{ext}"
        target_path = os.path.join(converted_root, target_filename)

        # 2. Check Cache / Convert
        if os.path.exists(target_path):
            print(f"[Luna] Found cached model: {target_filename}")
        else:
            print(f"[Luna] Cache miss. Converting {ckpt_name} to {precision}...")
            if is_gguf:
                self._convert_to_gguf(src_path, target_path, precision)
            else:
                self._convert_to_safetensors(src_path, target_path, precision, strip_components)

        # 3. Load the Result
        print(f"[Luna] Loading {target_path}...")
        
        if is_gguf:
            return self._load_gguf(target_path)
        else:
            return self._load_safetensors(target_path)

    def _convert_to_safetensors(self, src, dst, precision, strip):
        # Determine torch dtype
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp8_e4m3fn": torch.float8_e4m3fn
        }
        target_dtype = dtype_map.get(precision, torch.float16)

        state_dict = load_file(src)
        new_dict = {}

        for key, tensor in tqdm(state_dict.items(), desc=f"Converting to {precision}"):
            # Filtering Logic
            if strip:
                # Keep only UNet (model.diffusion_model)
                if not key.startswith("model.diffusion_model."):
                    continue
            
            # Casting Logic
            if tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                new_dict[key] = tensor.to(target_dtype)
            else:
                new_dict[key] = tensor
        
        print(f"[Luna] Saving to {dst}...")
        save_file(new_dict, dst)

    def _convert_to_gguf(self, src, dst, precision):
        # NOTE: This requires the external conversion script or library.
        # Since implementing the full quantizer in python node is risky, 
        # we will assume the user wants to use the `gguf` CLI or subprocess.
        # However, for a "Lazy" flow, we can use the library if installed.
        
        import subprocess
        import sys
        
        # Path to ComfyUI-GGUF converter if it exists in expected path
        # Adjust this path based on your installation!
        converter_script = os.path.join(folder_paths.base_path, "custom_nodes", "ComfyUI-GGUF", "tools", "convert.py")
        
        if not os.path.exists(converter_script):
            raise FileNotFoundError(f"Could not find GGUF converter at {converter_script}")

        quant_type = precision.replace("gguf_", "").upper() # Q8_0 or Q4_K_S
        
        cmd = [
            sys.executable, converter_script,
            "--in", src,
            "--out", dst,
            "--out-type", quant_type.lower(),
            "--unet-only" # We enforce UNet only for GGUF to match your architecture
        ]
        
        print(f"[Luna] Executing GGUF conversion: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    def _load_safetensors(self, path):
        # Standard ComfyUI Load
        out = comfy.sd.load_checkpoint_guess_config(path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return (out[0], out[1], out[2], path)

    def _load_gguf(self, path):
        if not HAS_GGUF_NODE:
            raise ImportError("ComfyUI-GGUF custom node is required to load .gguf files.")
        
        # Instantiate the GGUF Loader Logic
        loader = UnetLoaderGGUF()
        # The GGUF loader returns (MODEL,)
        model_tuple = loader.load_unet(path)
        
        # Return Dummy CLIP/VAE since GGUF is UNet only
        return (model_tuple[0], None, None, path)
```

### How this fits your workflow:
1.  **Format Handling:**
    *   If you pick **`fp8_e4m3fn`**, it creates a native `.safetensors` file. ComfyUI loads this using the standard loader, which now supports FP8 natively.
    *   If you pick **`gguf_Q4_K_S`**, it shells out to the script you already have installed, converts it, and then uses the logic from `ComfyUI-GGUF` to load it.
2.  **The "None" Return:**
    *   When stripping VAE/CLIP, the node returns `None` for those outputs.
    *   This forces you to rely on your **Luna Shared VAE/CLIP** nodes (or a separate specific loader), which aligns perfectly with your "Split Daemon" architecture.
3.  **One Node to Rule Them All:**
    *   You place this node in your workflow.
    *   On the **3090**, you set it to `gguf_Q8_0`.
    *   On the **5090**, you set it to `fp8_e4m3fn`.
    *   You select the *original* FP16 model name from the dropdown.
    *   You press Queue.
        *   *First time:* It pauses, converts, saves to NVMe, loads, generates.
        *   *Second time:* It instantly loads from NVMe.

This is the cleanest implementation of your "NVMe Cache" strategy.