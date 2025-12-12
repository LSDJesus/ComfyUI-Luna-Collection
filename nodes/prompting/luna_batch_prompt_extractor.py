"""
Luna Batch Prompt Extractor
Scans a directory of images and extracts complete metadata to JSON.
Preserves prompts, LoRAs, embeddings, and other generation parameters.
"""

import os
import json
import re
from PIL import Image
from PIL.ExifTags import IFD
from typing import Tuple, Dict, List, Any, Optional

try:
    import folder_paths
    HAS_FOLDER_PATHS = True
except ImportError:
    HAS_FOLDER_PATHS = False


class LunaBatchPromptExtractor:
    """
    Scan a directory of images and extract complete metadata to JSON.
    
    Each image entry includes:
    - image_name, image_path
    - positive_prompt, negative_prompt
    - loras (list of {name, weight})
    - embeddings (list of names)
    - Other metadata if available (steps, cfg, sampler, etc.)
    
    Supports multiple metadata formats:
    - ComfyUI (workflow JSON in PNG metadata)
    - A1111/Forge (parameters in PNG text)
    - InvokeAI, NovelAI, etc.
    """
    
    CATEGORY = "Luna/Utilities"
    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("status", "images_scanned", "images_extracted")
    FUNCTION = "extract_metadata"
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_directory": ("STRING", {
                    "default": "",
                    "tooltip": "Directory containing images with metadata"
                }),
                "output_file": ("STRING", {
                    "default": "prompts_metadata.json",
                    "tooltip": "JSON filename to save metadata"
                }),
                "save_to_input_dir": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Save to ComfyUI input directory (for use with Luna Batch Prompt Loader)"
                }),
                "output_directory": ("STRING", {
                    "default": "",
                    "tooltip": "Custom output directory (only used if save_to_input_dir is False)"
                }),
                "overwrite": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Overwrite existing JSON file"
                }),
                "include_path": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include full image path in JSON"
                }),
            }
        }
    
    def parse_loras_from_prompt(self, prompt: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract LoRA tags from prompt and return cleaned prompt + LoRA list
        
        Supports formats:
        - <lora:name:weight> - single weight used for both model and clip
        - <lora:name:model_weight:clip_weight> - separate weights
        """
        loras = []
        
        # Match <lora:name:weight> or <lora:name:model_weight:clip_weight> format
        lora_pattern = r'<lora:([^:>]+):([0-9.]+)(?::([0-9.]+))?>'
        matches = re.finditer(lora_pattern, prompt)
        
        for match in matches:
            name = match.group(1)
            model_strength = float(match.group(2))
            # If clip_strength specified, use it; otherwise use model_strength for both
            clip_strength = float(match.group(3)) if match.group(3) else model_strength
            
            loras.append({
                "name": name,
                "model_strength": model_strength,
                "clip_strength": clip_strength
            })
        
        # Remove LoRA tags from prompt
        cleaned_prompt = re.sub(lora_pattern, '', prompt).strip()
        # Clean up multiple spaces
        cleaned_prompt = re.sub(r'\s+', ' ', cleaned_prompt)
        
        return cleaned_prompt, loras
    
    def parse_embeddings_from_prompt(self, prompt: str) -> Tuple[str, List[str]]:
        """Extract embedding tags and return cleaned prompt + embedding list"""
        embeddings = []
        
        # Match common embedding formats: embedding:name or (embedding:name)
        embedding_pattern = r'(?:embedding:|^|\s)([a-zA-Z0-9_-]+(?:\.pt|\.safetensors)?)'
        
        # Simple heuristic: look for patterns that look like embeddings
        # This is imperfect without knowing which embeddings are installed
        # So we'll just return the prompt as-is and let user manually track embeddings
        
        return prompt, embeddings
    
    def extract_comfyui_metadata(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Extract full metadata from ComfyUI workflow"""
        try:
            img = Image.open(image_path)
            metadata = {
                "positive_prompt": "",
                "negative_prompt": "",
                "loras": [],
                "embeddings": [],
                "extra": {}
            }
            
            prompt_data = None
            workflow_data = None
            
            # ComfyUI stores 'prompt' (executed graph) and 'workflow' (full graph)
            if 'prompt' in img.info:
                prompt_data = json.loads(img.info['prompt'])
            if 'workflow' in img.info:
                workflow_data = json.loads(img.info['workflow'])
            
            if not prompt_data and not workflow_data:
                return None
            
            # Extract from prompt data (node_id -> node_data dict)
            if prompt_data and isinstance(prompt_data, dict):
                positive_texts = []
                negative_texts = []
                
                for node_id, node in prompt_data.items():
                    class_type = node.get('class_type', '')
                    inputs = node.get('inputs', {})
                    
                    # CLIP Text Encode nodes
                    if 'CLIPTextEncode' in class_type:
                        text = inputs.get('text', '')
                        if text:
                            # Check node title or connections to determine pos/neg
                            # Heuristic: check if 'negative' appears in any connected node names
                            positive_texts.append(text)
                    
                    # LoRA Loader nodes - ComfyUI has model_weight and clip_weight
                    if 'LoraLoader' in class_type or 'LoRALoader' in class_type:
                        lora_name = inputs.get('lora_name', '')
                        model_strength = float(inputs.get('strength_model', 1.0))
                        clip_strength = float(inputs.get('strength_clip', 1.0))
                        
                        if lora_name:
                            metadata["loras"].append({
                                "name": lora_name,
                                "model_strength": model_strength,
                                "clip_strength": clip_strength
                            })
                    
                    # KSampler nodes - extract seed
                    if 'KSampler' in class_type or 'Sampler' in class_type:
                        seed_val = inputs.get('seed', inputs.get('noise_seed', None))
                        if seed_val is not None:
                            metadata["extra"]["seed"] = int(seed_val)
                
                # Use longest text as positive (common heuristic)
                if positive_texts:
                    positive_texts.sort(key=len, reverse=True)
                    metadata["positive_prompt"] = positive_texts[0]
                    if len(positive_texts) > 1:
                        # Second longest is likely negative
                        metadata["negative_prompt"] = positive_texts[1]
            
            # Also check workflow for node titles to better identify pos/neg
            if workflow_data and isinstance(workflow_data, dict):
                nodes = workflow_data.get('nodes', [])
                if isinstance(nodes, list):
                    for node in nodes:
                        title = str(node.get('title', '')).lower()
                        widgets = node.get('widgets_values', [])
                        node_type = str(node.get('type', ''))
                        
                        # If title contains 'negative', it's the negative prompt
                        if 'negative' in title and 'CLIPTextEncode' in node_type:
                            if widgets and isinstance(widgets[0], str):
                                metadata["negative_prompt"] = widgets[0]
                        elif 'positive' in title and 'CLIPTextEncode' in node_type:
                            if widgets and isinstance(widgets[0], str):
                                metadata["positive_prompt"] = widgets[0]
            
            # Parse LoRAs from prompts (in case of inline <lora:> tags)
            metadata["positive_prompt"], loras_pos = self.parse_loras_from_prompt(metadata["positive_prompt"])
            metadata["negative_prompt"], loras_neg = self.parse_loras_from_prompt(metadata["negative_prompt"])
            
            # Merge LoRA lists (avoid duplicates by name)
            for lora in loras_pos + loras_neg:
                if not any(l["name"] == lora["name"] for l in metadata["loras"]):
                    metadata["loras"].append(lora)
            
            # Only return if we found something
            if metadata["positive_prompt"] or metadata["negative_prompt"] or metadata["loras"]:
                return metadata
            
            return None
            
        except:
            return None
    
    def extract_a1111_metadata(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Extract metadata from A1111/Forge parameters
        
        A1111 format:
        positive prompt (may be multi-line)
        Negative prompt: negative prompt (may be multi-line)
        Steps: X, Sampler: Y, CFG scale: Z, Seed: N, Size: WxH, ...
        
        Everything before 'Negative prompt:' is positive.
        Everything between 'Negative prompt:' and 'Steps:' is negative.
        """
        try:
            img = Image.open(image_path)
            metadata = {
                "positive_prompt": "",
                "negative_prompt": "",
                "loras": [],
                "embeddings": [],
                "extra": {}
            }
            
            # A1111 stores in 'parameters' key
            if 'parameters' not in img.info:
                return None
            
            params = img.info['parameters']
            
            # Find the positions of key markers
            neg_marker = "Negative prompt:"
            steps_marker = "Steps:"
            
            neg_pos = params.find(neg_marker)
            steps_pos = params.find(steps_marker)
            
            # Extract positive prompt (everything before Negative prompt: or Steps:)
            if neg_pos != -1:
                metadata["positive_prompt"] = params[:neg_pos].strip()
            elif steps_pos != -1:
                metadata["positive_prompt"] = params[:steps_pos].strip()
            else:
                # No markers found, treat entire thing as positive prompt
                metadata["positive_prompt"] = params.strip()
            
            # Extract negative prompt (between Negative prompt: and Steps:)
            if neg_pos != -1:
                neg_start = neg_pos + len(neg_marker)
                if steps_pos != -1 and steps_pos > neg_pos:
                    metadata["negative_prompt"] = params[neg_start:steps_pos].strip()
                else:
                    # No Steps marker, take everything after Negative prompt:
                    metadata["negative_prompt"] = params[neg_start:].strip()
            
            # Extract other parameters (after Steps:)
            if steps_pos != -1:
                params_section = params[steps_pos:]
                
                # Parse key: value pairs separated by commas
                # Handle multi-value keys like "Lora hashes" that contain colons
                param_pattern = r'([A-Za-z][A-Za-z0-9 _-]*?):\s*([^,]+(?:,(?![A-Za-z][A-Za-z0-9 _-]*?:)[^,]*)*)'
                
                for match in re.finditer(param_pattern, params_section):
                    key = match.group(1).strip()
                    value = match.group(2).strip()
                    key_lower = key.lower()
                    
                    # Store all extra parameters
                    metadata["extra"][key_lower] = value
                    
                    # Special handling for seed
                    if key_lower == 'seed':
                        try:
                            metadata["extra"]["seed"] = int(value)
                        except ValueError:
                            pass
                    
                    # Special handling for size -> extract width/height
                    if key_lower == 'size':
                        size_match = re.match(r'(\d+)x(\d+)', value)
                        if size_match:
                            metadata["extra"]["width"] = int(size_match.group(1))
                            metadata["extra"]["height"] = int(size_match.group(2))
            
            # Parse LoRAs from prompts
            metadata["positive_prompt"], loras_pos = self.parse_loras_from_prompt(metadata["positive_prompt"])
            metadata["negative_prompt"], loras_neg = self.parse_loras_from_prompt(metadata["negative_prompt"])
            metadata["loras"] = loras_pos + loras_neg
            
            # Remove duplicate LoRAs
            seen = set()
            unique_loras = []
            for lora in metadata["loras"]:
                if lora["name"] not in seen:
                    seen.add(lora["name"])
                    unique_loras.append(lora)
            metadata["loras"] = unique_loras
            
            return metadata
        except Exception as e:
            print(f"[LunaBatchPromptExtractor] Error extracting A1111 metadata from {image_path}: {e}")
            return None
    
    def extract_generic_metadata(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Try to extract metadata from any text metadata including EXIF"""
        try:
            img = Image.open(image_path)
            metadata = {
                "positive_prompt": "",
                "negative_prompt": "",
                "loras": [],
                "embeddings": [],
                "extra": {}
            }
            
            # Check all text fields for prompt-like content
            for key, value in img.info.items():
                if isinstance(key, str) and isinstance(value, str):
                    # Look for common prompt indicators
                    if 'prompt' in key.lower():
                        if 'negative' in key.lower() or 'neg' in key.lower():
                            metadata["negative_prompt"] = value
                        else:
                            metadata["positive_prompt"] = value
                    else:
                        # Store other metadata
                        metadata["extra"][key] = value
            
            # Parse LoRAs if found in prompts
            if metadata["positive_prompt"]:
                metadata["positive_prompt"], loras = self.parse_loras_from_prompt(metadata["positive_prompt"])
                metadata["loras"].extend(loras)
            
            if metadata["negative_prompt"]:
                metadata["negative_prompt"], loras = self.parse_loras_from_prompt(metadata["negative_prompt"])
                metadata["loras"].extend(loras)
            
            return metadata if (metadata["positive_prompt"] or metadata["negative_prompt"]) else None
            
        except:
            return None
    
    def decode_exif_user_comment(self, comment) -> str:
        """Decode EXIF UserComment field which has an 8-byte charset prefix.
        
        Format: 8-byte charset identifier + data
        Charsets: 'ASCII\x00\x00\x00', 'JIS\x00\x00\x00\x00\x00', 'UNICODE\x00', or 8 null bytes (undefined)
        
        Note: Despite the EXIF spec, some tools (like CivitAI) write UNICODE data as UTF-16BE,
        not UTF-16LE. We detect this by checking if the first byte after the marker is null.
        """
        if comment is None:
            return ""
        
        # If already a string, return it
        if isinstance(comment, str):
            return comment.strip().strip('\x00')
        
        if not isinstance(comment, bytes):
            return str(comment)
        
        # EXIF UserComment has 8-byte charset prefix
        if len(comment) <= 8:
            return ""
        
        charset_marker = comment[:8]
        data = comment[8:]
        
        # Try to decode based on charset marker
        try:
            if charset_marker.startswith(b'ASCII'):
                return data.decode('ascii', errors='ignore').strip().strip('\x00')
            elif charset_marker.startswith(b'UNICODE'):
                # UNICODE can be UTF-16LE or UTF-16BE depending on the writer
                # Detect by checking if first byte is null (BE) or second byte is null (LE)
                if len(data) >= 2:
                    if data[0] == 0 and data[1] != 0:
                        # First byte is null, second is not -> UTF-16BE (e.g., \x00m for 'm')
                        decoded = data.decode('utf-16-be', errors='ignore')
                    elif data[0] != 0 and data[1] == 0:
                        # First byte is char, second is null -> UTF-16LE (e.g., m\x00 for 'm')
                        decoded = data.decode('utf-16-le', errors='ignore')
                    else:
                        # Try both, pick the one that looks like ASCII text
                        try:
                            be = data.decode('utf-16-be', errors='ignore')
                            le = data.decode('utf-16-le', errors='ignore')
                            # Prefer the one with more printable ASCII chars
                            be_ascii = sum(1 for c in be[:50] if 32 <= ord(c) <= 126)
                            le_ascii = sum(1 for c in le[:50] if 32 <= ord(c) <= 126)
                            decoded = be if be_ascii >= le_ascii else le
                        except:
                            decoded = data.decode('utf-16', errors='ignore')
                else:
                    decoded = data.decode('utf-16', errors='ignore')
                return decoded.strip().strip('\x00')
            elif charset_marker.startswith(b'JIS'):
                return data.decode('shift_jis', errors='ignore').strip().strip('\x00')
            elif charset_marker == b'\x00\x00\x00\x00\x00\x00\x00\x00':
                # Undefined - try UTF-8, then latin-1
                try:
                    return data.decode('utf-8').strip().strip('\x00')
                except:
                    return data.decode('latin-1', errors='ignore').strip().strip('\x00')
            else:
                # Unknown marker - might not have a proper prefix
                # Try decoding the whole thing as UTF-8 or ASCII
                try:
                    return comment.decode('utf-8').strip().strip('\x00')
                except:
                    return comment.decode('latin-1', errors='ignore').strip().strip('\x00')
        except Exception as e:
            # Last resort - try to decode as-is
            try:
                return comment.decode('utf-8', errors='ignore').strip().strip('\x00')
            except:
                return comment.decode('latin-1', errors='ignore').strip().strip('\x00')
    
    def extract_exif_metadata(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Extract metadata from EXIF data (JPEG, WebP, TIFF)
        
        Many AI tools store prompts in EXIF UserComment or ImageDescription fields.
        Some use custom EXIF tags or XMP metadata.
        """
        try:
            img = Image.open(image_path)
            metadata = {
                "positive_prompt": "",
                "negative_prompt": "",
                "loras": [],
                "embeddings": [],
                "extra": {}
            }
            
            # Try to get EXIF data
            exif_data = None
            # Use getexif() which is the modern PIL API
            try:
                exif_data = img.getexif()
                if not exif_data:
                    exif_data = None
            except Exception:
                pass
            
            # Fallback: try legacy _getexif if available
            if exif_data is None:
                _getexif = getattr(img, '_getexif', None)
                if _getexif is not None:
                    try:
                        exif_data = _getexif()
                    except Exception:
                        pass
            
            if not exif_data:
                return None
            
            # Common EXIF tags that might contain prompts
            # 270 = ImageDescription
            # 37510 = UserComment  
            # 40092 = XPComment (Windows)
            # 40091 = XPTitle (Windows)
            
            prompt_text = ""
            
            # Check ImageDescription (tag 270)
            if 270 in exif_data:
                desc = exif_data[270]
                if isinstance(desc, bytes):
                    desc = desc.decode('utf-8', errors='ignore')
                if desc and len(desc) > 10:
                    prompt_text = desc
            
            # Check UserComment (tag 37510) - uses helper for proper charset decoding
            if 37510 in exif_data and not prompt_text:
                comment = self.decode_exif_user_comment(exif_data[37510])
                if comment and len(comment) > 10:
                    prompt_text = comment
            
            # Check for IFD.Exif sub-IFD (newer PIL approach)
            try:
                exif_ifd = img.getexif().get_ifd(IFD.Exif)
                if exif_ifd:
                    # 37510 = UserComment in Exif IFD
                    if 37510 in exif_ifd:
                        comment = self.decode_exif_user_comment(exif_ifd[37510])
                        if comment and len(comment) > 10:
                            prompt_text = comment
            except:
                pass
            
            if not prompt_text:
                return None
            
            # Clean up the prompt text
            prompt_text = prompt_text.strip().strip('\x00')
            
            # Check if it looks like A1111 format
            if "Negative prompt:" in prompt_text or "Steps:" in prompt_text:
                # Parse as A1111 format
                neg_marker = "Negative prompt:"
                steps_marker = "Steps:"
                
                neg_pos = prompt_text.find(neg_marker)
                steps_pos = prompt_text.find(steps_marker)
                
                if neg_pos != -1:
                    metadata["positive_prompt"] = prompt_text[:neg_pos].strip()
                    neg_start = neg_pos + len(neg_marker)
                    if steps_pos != -1 and steps_pos > neg_pos:
                        metadata["negative_prompt"] = prompt_text[neg_start:steps_pos].strip()
                    else:
                        metadata["negative_prompt"] = prompt_text[neg_start:].strip()
                elif steps_pos != -1:
                    metadata["positive_prompt"] = prompt_text[:steps_pos].strip()
                else:
                    metadata["positive_prompt"] = prompt_text
            else:
                # Just treat the whole thing as positive prompt
                metadata["positive_prompt"] = prompt_text
            
            # Parse LoRAs from prompts
            if metadata["positive_prompt"]:
                metadata["positive_prompt"], loras = self.parse_loras_from_prompt(metadata["positive_prompt"])
                metadata["loras"].extend(loras)
            
            if metadata["negative_prompt"]:
                metadata["negative_prompt"], loras = self.parse_loras_from_prompt(metadata["negative_prompt"])
                metadata["loras"].extend(loras)
            
            return metadata if (metadata["positive_prompt"] or metadata["negative_prompt"]) else None
            
        except Exception as e:
            print(f"[LunaBatchPromptExtractor] Error extracting EXIF from {image_path}: {e}")
            return None
    
    def extract_metadata(
        self,
        image_directory: str,
        output_file: str,
        save_to_input_dir: bool,
        output_directory: str,
        overwrite: bool,
        include_path: bool
    ) -> Tuple[str, int, int]:
        """Extract complete metadata from all images in directory and save as JSON"""
        
        if not os.path.isdir(image_directory):
            return (f"Error: Directory not found: {image_directory}", 0, 0)
        
        # Determine output directory
        if save_to_input_dir and HAS_FOLDER_PATHS:
            # Save to ComfyUI input directory for easy loading
            output_directory = folder_paths.get_input_directory()
        elif not output_directory or not os.path.isdir(output_directory):
            # Default to image directory
            output_directory = image_directory
        
        output_path = os.path.join(output_directory, output_file)
        
        # Check if file exists and overwrite is False
        if not overwrite and os.path.exists(output_path):
            return ("Error: Output file exists and overwrite=False", 0, 0)
        
        # Scan directory for images
        supported_exts = {'.png', '.jpg', '.jpeg', '.webp'}
        image_files = []
        
        for filename in os.listdir(image_directory):
            ext = os.path.splitext(filename)[1].lower()
            if ext in supported_exts:
                image_files.append(os.path.join(image_directory, filename))
        
        if not image_files:
            return (f"No images found in {image_directory}", 0, 0)
        
        # Extract metadata from all images
        metadata_list = []
        images_scanned = 0
        metadata_extracted = 0
        no_metadata_files = []
        
        for image_path in image_files:
            images_scanned += 1
            
            # Try different extraction methods in order of specificity
            metadata = self.extract_comfyui_metadata(image_path)
            if not metadata:
                metadata = self.extract_a1111_metadata(image_path)
            if not metadata:
                metadata = self.extract_exif_metadata(image_path)
            if not metadata:
                metadata = self.extract_generic_metadata(image_path)
            
            # Skip if no metadata found
            if not metadata:
                no_metadata_files.append(os.path.basename(image_path))
                continue
            
            # Add image file information
            metadata["image_name"] = os.path.basename(image_path)
            metadata["image_path"] = os.path.abspath(image_path)
            
            # Extract image dimensions
            try:
                with Image.open(image_path) as img:
                    metadata["width"] = img.width
                    metadata["height"] = img.height
            except Exception as e:
                print(f"[LunaBatchPromptExtractor] Could not get dimensions for {image_path}: {e}")
                metadata["width"] = 0
                metadata["height"] = 0
            
            metadata_list.append(metadata)
            metadata_extracted += 1
        
        # Write to JSON file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_list, f, indent=2, ensure_ascii=False)
            
            status_lines = [
                "Success!",
                f"Scanned: {images_scanned} images",
                f"Extracted: {metadata_extracted} with metadata",
                f"Saved to: {output_path}"
            ]
            
            if no_metadata_files:
                status_lines.append(f"\nNo metadata found in {len(no_metadata_files)} files:")
                for fname in no_metadata_files[:10]:  # Limit to first 10
                    status_lines.append(f"  - {fname}")
                if len(no_metadata_files) > 10:
                    status_lines.append(f"  ... and {len(no_metadata_files) - 10} more")
            
            return ("\n".join(status_lines), images_scanned, metadata_extracted)
            
        except Exception as e:
            return (f"Error writing JSON file: {e}", images_scanned, 0)


class LunaBatchPromptLoader:
    """
    Load metadata from JSON file created by LunaBatchPromptExtractor.
    Returns prompts and LORA_STACK compatible with Apply LoRA Stack nodes.
    
    Features:
    - Sequential: index auto-increments via control_after_generate
    - Random: picks a random entry (use randomize on index widget)
    - Option to append LoRAs as inline <lora:name:model:clip> tags
    - Option to validate LoRAs/embeddings exist before including them
    - Resolves full paths for LoRAs in subdirectories
    - Outputs seed from metadata for reproducibility
    - Outputs list_complete flag for model switching workflows
    """
    
    CATEGORY = "Luna/Utilities"
    RETURN_TYPES = ("STRING", "STRING", "LORA_STACK", "INT", "INT", "INT", "BOOLEAN", "INT", "INT")
    RETURN_NAMES = ("positive", "negative", "lora_stack", "seed", "current_index", "total_entries", "list_complete", "width", "height")
    FUNCTION = "load_metadata"
    
    # Cache for discovered LoRAs/embeddings (name -> relative path)
    _lora_cache: Dict[str, List[str]] = {}
    _embedding_cache: Dict[str, List[str]] = {}
    _cache_initialized = False
    
    # Cache for JSON file entry counts (file_path -> entry_count)
    _json_entry_cache: Dict[str, int] = {}
    
    @classmethod
    def _get_json_files(cls) -> List[str]:
        """Get list of JSON files from ComfyUI input directory"""
        if not HAS_FOLDER_PATHS:
            return []
        
        try:
            input_dir = folder_paths.get_input_directory()
            files = []
            for f in os.listdir(input_dir):
                if f.endswith('.json') and os.path.isfile(os.path.join(input_dir, f)):
                    files.append(f)
            return sorted(files) if files else ["No JSON files found"]
        except Exception as e:
            print(f"[LunaBatchPromptLoader] Error scanning input directory: {e}")
            return ["Error scanning directory"]
    
    @classmethod
    def _get_json_entry_count(cls, json_file: str) -> int:
        """Get the number of entries in a JSON file (cached)"""
        if not HAS_FOLDER_PATHS:
            return 0
        
        try:
            file_path = folder_paths.get_annotated_filepath(json_file)
            
            # Check cache first (use mtime to invalidate)
            if os.path.isfile(file_path):
                mtime = os.path.getmtime(file_path)
                cache_key = f"{file_path}:{mtime}"
                
                if cache_key in cls._json_entry_cache:
                    return cls._json_entry_cache[cache_key]
                
                # Load and count entries
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        count = len(data)
                        cls._json_entry_cache[cache_key] = count
                        return count
        except Exception:
            pass
        
        return 0
    
    @classmethod
    def INPUT_TYPES(cls):
        json_files = cls._get_json_files()
        
        return {
            "required": {
                "json_file": (json_files, {
                    "tooltip": "Select a JSON metadata file from the input directory. Use 'choose file to upload' to add new files."
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "Current index in the prompt list. Use increment/randomize to step through entries."
                }),
                "lora_output": (["stack_only", "inline_only", "both"], {
                    "default": "stack_only",
                    "tooltip": "stack_only: LORA_STACK output only\ninline_only: Append <lora:> tags to prompt\nboth: Both outputs"
                }),
                "lora_validation": (["include_all", "only_existing"], {
                    "default": "include_all",
                    "tooltip": "include_all: Include all LoRAs from metadata\nonly_existing: Only include LoRAs that exist on disk"
                }),
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, json_file, **kwargs):
        """Check if the JSON file has been modified"""
        if not HAS_FOLDER_PATHS:
            return float("nan")
        
        try:
            file_path = folder_paths.get_annotated_filepath(json_file)
            if os.path.isfile(file_path):
                # Return mtime so node updates when file changes
                return os.path.getmtime(file_path)
        except Exception:
            pass
        
        return float("nan")
    
    @classmethod
    def VALIDATE_INPUTS(cls, json_file, **kwargs):
        """Validate that the selected JSON file exists"""
        if json_file in ["No JSON files found", "Error scanning directory"]:
            return f"No valid JSON file selected: {json_file}"
        
        if not HAS_FOLDER_PATHS:
            return "folder_paths module not available"
        
        try:
            file_path = folder_paths.get_annotated_filepath(json_file)
            if not os.path.isfile(file_path):
                return f"JSON file not found: {json_file}"
        except Exception as e:
            return f"Error validating file: {e}"
        
        return True
    
    @classmethod
    def _build_cache(cls):
        """Scan loras and embeddings directories to build filename -> path cache"""
        if cls._cache_initialized or not HAS_FOLDER_PATHS:
            return
        
        cls._lora_cache = {}
        cls._embedding_cache = {}
        
        # Scan LoRA directories
        try:
            lora_paths = folder_paths.get_folder_paths("loras")
            for base_path in lora_paths:
                if not os.path.isdir(base_path):
                    continue
                for root, dirs, files in os.walk(base_path):
                    for filename in files:
                        if filename.endswith(('.safetensors', '.pt', '.ckpt', '.bin')):
                            # Get relative path from base
                            full_path = os.path.join(root, filename)
                            rel_path = os.path.relpath(full_path, base_path)
                            # Store by filename (without extension) for matching
                            name_no_ext = os.path.splitext(filename)[0]
                            if name_no_ext not in cls._lora_cache:
                                cls._lora_cache[name_no_ext] = []
                            cls._lora_cache[name_no_ext].append(rel_path)
        except Exception as e:
            print(f"[LunaBatchPromptLoader] Error scanning loras: {e}")
        
        # Scan embedding directories
        try:
            embedding_paths = folder_paths.get_folder_paths("embeddings")
            for base_path in embedding_paths:
                if not os.path.isdir(base_path):
                    continue
                for root, dirs, files in os.walk(base_path):
                    for filename in files:
                        if filename.endswith(('.safetensors', '.pt', '.bin')):
                            full_path = os.path.join(root, filename)
                            rel_path = os.path.relpath(full_path, base_path)
                            name_no_ext = os.path.splitext(filename)[0]
                            if name_no_ext not in cls._embedding_cache:
                                cls._embedding_cache[name_no_ext] = []
                            cls._embedding_cache[name_no_ext].append(rel_path)
        except Exception as e:
            print(f"[LunaBatchPromptLoader] Error scanning embeddings: {e}")
        
        cls._cache_initialized = True
        print(f"[LunaBatchPromptLoader] Cached {len(cls._lora_cache)} LoRAs, {len(cls._embedding_cache)} embeddings")
    
    def resolve_lora_path(self, lora_name: str) -> Optional[str]:
        """Resolve a LoRA name to its full relative path
        
        If multiple matches exist (same name in different folders), returns the first found.
        Returns None if not found.
        """
        self._build_cache()
        
        # First check if it's already a path (contains / or \)
        if '/' in lora_name or '\\' in lora_name:
            # Already a path - validate it exists
            name_no_ext = os.path.splitext(lora_name)[0]
            # Check if this exact path exists in any lora folder
            if HAS_FOLDER_PATHS:
                try:
                    full_path = folder_paths.get_full_path("loras", lora_name)
                    if full_path and os.path.exists(full_path):
                        return lora_name
                except:
                    pass
            return None
        
        # Strip extension if present
        name_no_ext = os.path.splitext(lora_name)[0]
        
        # Look up in cache
        if name_no_ext in self._lora_cache:
            paths = self._lora_cache[name_no_ext]
            if len(paths) == 1:
                return paths[0]
            else:
                # Multiple matches - return first but log warning
                print(f"[LunaBatchPromptLoader] Multiple LoRAs named '{name_no_ext}': {paths}. Using: {paths[0]}")
                return paths[0]
        
        return None
    
    def resolve_embedding_path(self, embedding_name: str) -> Optional[str]:
        """Resolve an embedding name to its full relative path"""
        self._build_cache()
        
        name_no_ext = os.path.splitext(embedding_name)[0]
        
        if name_no_ext in self._embedding_cache:
            paths = self._embedding_cache[name_no_ext]
            if len(paths) >= 1:
                return paths[0]
        
        return None
    
    def load_metadata(
        self,
        json_file: str,
        index: int,
        lora_output: str,
        lora_validation: str
    ) -> Tuple[str, str, List[Tuple[str, float, float]], int, int, int, bool, int, int]:
        """Load metadata entry from JSON file
        
        Returns:
            - positive prompt (str) - may include inline <lora:> tags
            - negative prompt (str)  
            - lora_stack: List[Tuple[lora_name, model_strength, clip_strength]]
            - seed (int) - extracted seed from metadata, or 0 if not found
            - current_index (int) - the actual index used
            - total_entries (int) - total entries in the JSON
            - list_complete (bool) - True if this is the last entry in the list
            - width (int) - original image width
            - height (int) - original image height
        """
        
        # Resolve file path from input directory
        if HAS_FOLDER_PATHS:
            file_path = folder_paths.get_annotated_filepath(json_file)
        else:
            file_path = json_file
        
        if not os.path.isfile(file_path):
            return (f"Error: {json_file} not found", "", [], 0, 0, 0, False, 0, 0)
        
        try:
            # Load JSON metadata
            with open(file_path, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)
            
            if not metadata_list or not isinstance(metadata_list, list):
                return ("Error: JSON file is empty or invalid", "", [], 0, 0, 0, False, 0, 0)
            
            total = len(metadata_list)
            
            # Use index directly - control_after_generate handles increment/randomize
            current_index = index % total
            
            # Check if this is the last entry (index + 1 would wrap around)
            list_complete = (current_index >= total - 1)
            
            # Get entry
            entry = metadata_list[current_index]
            
            pos_prompt = entry.get("positive_prompt", "")
            neg_prompt = entry.get("negative_prompt", "")
            
            # Extract dimensions
            img_width = entry.get("width", 0)
            img_height = entry.get("height", 0)
            
            # Extract seed from metadata
            extra = entry.get("extra", {})
            extracted_seed = extra.get("seed", 0)
            if isinstance(extracted_seed, str):
                try:
                    extracted_seed = int(extracted_seed)
                except ValueError:
                    extracted_seed = 0
            
            # Process LoRAs
            loras = entry.get("loras", [])
            lora_stack = []
            inline_lora_strings = []
            
            for lora in loras:
                name = lora.get("name", "")
                model_strength = float(lora.get("model_strength", lora.get("weight", 1.0)))
                clip_strength = float(lora.get("clip_strength", model_strength))
                
                # Resolve path and validate if needed
                resolved_name = name
                if lora_validation == "only_existing":
                    resolved_path = self.resolve_lora_path(name)
                    if resolved_path is None:
                        # Skip this LoRA - doesn't exist
                        print(f"[LunaBatchPromptLoader] Skipping LoRA '{name}' - not found on disk")
                        continue
                    resolved_name = resolved_path
                
                # Add to stack (unless inline_only)
                if lora_output != "inline_only":
                    lora_stack.append((resolved_name, model_strength, clip_strength))
                
                # Build inline string (unless stack_only)
                if lora_output != "stack_only":
                    # Format: <lora:name:model:clip> or <lora:name:weight> if same
                    if abs(model_strength - clip_strength) < 0.001:
                        inline_lora_strings.append(f"<lora:{resolved_name}:{model_strength:.2f}>")
                    else:
                        inline_lora_strings.append(f"<lora:{resolved_name}:{model_strength:.2f}:{clip_strength:.2f}>")
            
            # Append inline LoRAs to positive prompt if needed
            if inline_lora_strings and lora_output in ("inline_only", "both"):
                pos_prompt = pos_prompt.strip() + " " + " ".join(inline_lora_strings)
            
            return (pos_prompt, neg_prompt, lora_stack, extracted_seed, current_index, total, list_complete, img_width, img_height)
            
        except Exception as e:
            return (f"Error loading metadata: {e}", "", [], 0, 0, 0, False, 0, 0)


class LunaDimensionScaler:
    """
    Scale input dimensions to native model resolutions.
    
    Takes width/height and reduces the larger dimension to the model's
    max native size, then scales the smaller dimension proportionally.
    Output dimensions are rounded to the nearest multiple of 8 (required for latent space).
    """
    
    CATEGORY = "Luna/Utilities"
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "scale_dimensions"
    
    # Native resolutions for different model types
    # These are the max dimensions models were trained on
    MODEL_NATIVE_SIZES = {
        "SD 1.5": 512,
        "SD 2.1": 768,
        "SDXL": 1024,
        "SD 3.5": 1024,
        "Flux": 1024,
        "Illustrious": 1024,
        "Pony": 1024,
        "Cascade": 1024,
        "Custom": 1024,
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "Input width (from Luna Batch Prompt Loader or manual)"
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "Input height (from Luna Batch Prompt Loader or manual)"
                }),
                "model_type": (list(cls.MODEL_NATIVE_SIZES.keys()), {
                    "default": "SDXL",
                    "tooltip": "Model type determines the max native resolution"
                }),
                "custom_max_size": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "Custom max size (only used when model_type is 'Custom')"
                }),
                "round_to": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Round output dimensions to nearest multiple (8 for latent, 64 for some models)"
                }),
            }
        }
    
    def scale_dimensions(
        self,
        width: int,
        height: int,
        model_type: str,
        custom_max_size: int,
        round_to: int
    ) -> Tuple[int, int]:
        """
        Scale dimensions so the larger side matches model's native max size.
        Smaller side is scaled proportionally, both rounded to nearest multiple.
        """
        # Get max size for model type
        if model_type == "Custom":
            max_size = custom_max_size
        else:
            max_size = self.MODEL_NATIVE_SIZES.get(model_type, 1024)
        
        # Handle edge cases
        if width <= 0 or height <= 0:
            return (max_size, max_size)
        
        # Determine which dimension is larger
        if width >= height:
            # Width is larger or equal - scale width to max, height proportionally
            if width <= max_size:
                # Already fits, just round
                new_width = width
                new_height = height
            else:
                scale = max_size / width
                new_width = max_size
                new_height = int(height * scale)
        else:
            # Height is larger - scale height to max, width proportionally
            if height <= max_size:
                # Already fits, just round
                new_width = width
                new_height = height
            else:
                scale = max_size / height
                new_height = max_size
                new_width = int(width * scale)
        
        # Round to nearest multiple
        new_width = max(round_to, round(new_width / round_to) * round_to)
        new_height = max(round_to, round(new_height / round_to) * round_to)
        
        return (new_width, new_height)


# Node registration
NODE_CLASS_MAPPINGS = {
    "LunaBatchPromptExtractor": LunaBatchPromptExtractor,
    "LunaBatchPromptLoader": LunaBatchPromptLoader,
    "LunaDimensionScaler": LunaDimensionScaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaBatchPromptExtractor": "Luna Batch Prompt Extractor",
    "LunaBatchPromptLoader": "Luna Batch Prompt Loader",
    "LunaDimensionScaler": "Luna Dimension Scaler",
}
