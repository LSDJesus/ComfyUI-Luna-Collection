"""
Luna SAM3 Detector - Semantic Object Detection & Planning

The "Site Surveyor" - identifies objects, classifies by hierarchy, and creates
the master plan for targeted refinement. Uses SAM3's Promptable Concept
Segmentation to find objects based on text descriptions.

Architecture:
- Supports up to 10 concept slots (expandable UI)
- Each concept has: name, prompt, layer, selection mode, max objects
- Outputs normalized coordinates (0.0-1.0) for resolution-independent detections
- Hierarchical layer system for sequential refinement (structural → detail)
- Uses Luna Daemon for shared SAM3 model across instances (3.4GB saved)

Workflow Integration:
    Image (1K draft) → Detector → Detection Data → Semantic Detailer
"""

import json
from typing import List, Dict, Tuple, Optional

import torch
import numpy as np
import comfy.model_management

# Import daemon SAM3 proxy
import sys
import os
luna_daemon_path = os.path.join(os.path.dirname(__file__), "..", "..", "luna_daemon")
if luna_daemon_path not in sys.path:
    sys.path.insert(0, luna_daemon_path)

from proxy import DaemonSAM3


# Custom data type for detection results (includes pre-encoded conditioning)
LUNA_DETECTION_PIPE = "LUNA_DETECTION_PIPE"


class LunaSAM3Detector:
    """
    Multi-concept object detector using SAM3.
    
    Finds objects in an image based on text descriptions, assigns them to
    hierarchical layers, and outputs normalized coordinates + conditioning
    for downstream refinement.
    
    Features:
    - Up to 10 configurable concept slots
    - Hierarchical layering (0=structural, 1+=details)
    - Multiple selection modes (largest, center, first N)
    - Normalized coordinates for resolution independence
    """
    
    CATEGORY = "Luna/Detailing"
    RETURN_TYPES = (LUNA_DETECTION_PIPE, "IMAGE")
    RETURN_NAMES = ("detection_pipe", "image_passthrough")
    FUNCTION = "detect"
    
    @classmethod
    def INPUT_TYPES(cls):
        # Selection mode options
        selection_modes = ["largest", "center", "first", "all"]
        
        inputs = {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image (preferably 1024px draft for speed)"
                }),
                "clip": ("CLIP", {
                    "tooltip": "CLIP model for encoding custom concept prompts"
                }),
                "positive": ("CONDITIONING", {
                    "tooltip": "Base positive conditioning (used when concept prompt is empty)"
                }),
                "negative": ("CONDITIONING", {
                    "tooltip": "Base negative conditioning (applied to all detections)"
                }),
                "sam3_model_name": ("STRING", {
                    "default": "sam3.safetensors",
                    "tooltip": "SAM3 model filename in models/sam3/"
                }),
                "device": (["cuda:0", "cuda:1", "cpu"], {
                    "default": "cuda:1",
                    "tooltip": "Device for SAM3 inference (use secondary GPU to free primary for UNet)"
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Minimum confidence score for detections"
                }),
                "padding_factor": ("FLOAT", {
                    "default": 1.15,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Context padding around detections (1.15 = 15% buffer)"
                }),
            },
            "optional": {}
        }
        
        # Add concept slots (1-10)
        for i in range(1, 11):
            inputs["optional"][f"concept_{i}_enabled"] = ("BOOLEAN", {
                "default": True if i == 1 else False,
                "tooltip": f"Enable concept slot {i}"
            })
            inputs["optional"][f"concept_{i}_name"] = ("STRING", {
                "default": "" if i > 1 else "face",
                "multiline": False,
                "tooltip": f"Concept {i}: Object to detect (e.g., 'face', 'hands', 'eyes')"
            })
            inputs["optional"][f"concept_{i}_prompt"] = ("STRING", {
                "default": "",
                "multiline": True,
                "tooltip": f"Concept {i}: Refinement prompt for this object"
            })
            inputs["optional"][f"concept_{i}_layer"] = ("INT", {
                "default": 0 if i == 1 else 1,
                "min": 0,
                "max": 5,
                "tooltip": f"Concept {i}: Layer for hierarchical processing (0=structural, 1+=details)"
            })
            inputs["optional"][f"concept_{i}_selection"] = (selection_modes, {
                "default": "largest",
                "tooltip": f"Concept {i}: How to select from multiple detections"
            })
            inputs["optional"][f"concept_{i}_max_objects"] = ("INT", {
                "default": 1,
                "min": 1,
                "max": 20,
                "tooltip": f"Concept {i}: Maximum objects to keep for this concept"
            })
        
        return inputs
    
    def detect(
        self,
        image: torch.Tensor,
        clip,
        positive,
        negative,
        sam3_model_name: str,
        device: str,
        confidence_threshold: float,
        padding_factor: float,
        **concept_kwargs
    ) -> Tuple[Dict, torch.Tensor]:
        """
        Detect objects using SAM3 and create detection plan.
        
        Args:
            image: Input image tensor [B, H, W, C] (ComfyUI format)
            sam3_model_name: Model filename
            confidence_threshold: Min confidence for detections
            padding_factor: Context padding multiplier
            **concept_kwargs: Concept slot parameters (concept_1_name, etc.)
            
        Returns:
            Tuple of (detection_data_dict, image_passthrough)
        """
        # Get image dimensions
        batch_size, height, width, channels = image.shape
        
        # Only process first image in batch
        if batch_size > 1:
            print(f"[LunaSAM3Detector] Warning: Batch size > 1, using first image only")
            image = image[0:1]
        
        # Parse concept configurations
        concepts = self._parse_concepts(concept_kwargs)
        
        if not concepts:
            print("[LunaSAM3Detector] No enabled concepts, returning empty detection data")
            return self._empty_detection_data(width, height), image
        
        print(f"[LunaSAM3Detector] Processing {len(concepts)} concepts")
        
        # Create SAM3 daemon proxy
        try:
            sam3_proxy = DaemonSAM3(model_name=sam3_model_name, device=device)
            if not sam3_proxy.load_model():
                print("[LunaSAM3Detector] Error: Failed to load SAM3 from daemon")
                return self._empty_detection_data(width, height), image
        except Exception as e:
            print(f"[LunaSAM3Detector] Error creating SAM3 proxy: {e}")
            return self._empty_detection_data(width, height), image
        
        # Convert ComfyUI image to PIL for SAM3
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        from PIL import Image
        pil_image = Image.fromarray(image_np)
        
        # Build batch prompts for all concepts at once (more efficient)
        batch_prompts = []
        concept_configs = {}  # Map label -> concept config for post-processing
        
        for concept in concepts:
            concept_name = concept["name"]
            batch_prompts.append({
                "prompt": concept_name,
                "threshold": confidence_threshold,
                "label": concept_name
            })
            concept_configs[concept_name] = concept
        
        print(f"[LunaSAM3Detector] Running batch detection for {len(batch_prompts)} concepts")
        
        # Run SAM3 batch grounding (reuses backbone features across prompts)
        try:
            results_by_label = sam3_proxy.ground_batch(
                image=pil_image,
                prompts=batch_prompts,
                default_threshold=confidence_threshold
            )
        except Exception as e:
            print(f"[LunaSAM3Detector] Error in batch detection: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_detection_data(width, height), image
        
        # Process results for each concept
        all_detections = []
        
        for concept_name, detections in results_by_label.items():
            concept = concept_configs.get(concept_name)
            if not concept:
                continue
            
            concept_prompt = concept["prompt"]
            layer = concept["layer"]
            selection_mode = concept["selection"]
            max_objects = concept["max_objects"]
            
            if not detections:
                print(f"[LunaSAM3Detector]   No detections for '{concept_name}'")
                continue
            
            # Apply selection logic
            selected = self._select_detections(
                detections=detections,
                mode=selection_mode,
                max_count=max_objects,
                image_size=(width, height)
            )
            
            print(f"[LunaSAM3Detector]   '{concept_name}': found {len(detections)}, selected {len(selected)}")
            
            # Normalize and store
            for det in selected:
                all_detections.append({
                    "concept": concept_name,
                    "prompt": concept_prompt,
                    "layer": layer,
                    "bbox_norm": self._normalize_bbox(det["bbox"], width, height),
                    "mask": det["mask"],
                    "confidence": det["confidence"]
                })
        
        # Encode prompts for all detections
        positive_batch = self._encode_prompts(clip, positive, all_detections)
        
        # Build detection pipe (no prompt strings in detections)
        detection_pipe = self._build_detection_pipe(
            detections=all_detections,
            positive_batch=positive_batch,
            image_size=(width, height),
            padding_factor=padding_factor
        )
        
        print(f"[LunaSAM3Detector] Complete: {len(all_detections)} total detections")
        
        return (detection_pipe, image)
    
    def _parse_concepts(self, kwargs: dict) -> List[Dict]:
        """Extract enabled concept configurations from kwargs."""
        concepts = []
        
        for i in range(1, 11):
            enabled = kwargs.get(f"concept_{i}_enabled", False)
            if not enabled:
                continue
            
            name = kwargs.get(f"concept_{i}_name", "").strip()
            if not name:
                continue
            
            concepts.append({
                "name": name,
                "prompt": kwargs.get(f"concept_{i}_prompt", "").strip(),
                "layer": kwargs.get(f"concept_{i}_layer", 0),
                "selection": kwargs.get(f"concept_{i}_selection", "largest"),
                "max_objects": kwargs.get(f"concept_{i}_max_objects", 1)
            })
        
        return concepts
    
    def _run_sam3_grounding(
        self,
        sam3_proxy: DaemonSAM3,
        image,  # PIL Image
        text_prompt: str,
        confidence_threshold: float
    ) -> List[Dict]:
        """
        Run SAM3 grounding detection via daemon proxy.
        
        Returns:
            List of dicts with keys: bbox, mask, confidence
        """
        try:
            # Call daemon proxy (handles serialization internally)
            detections = sam3_proxy.ground(
                image=image,
                text_prompt=text_prompt,
                threshold=confidence_threshold
            )
            
            return detections
            
        except Exception as e:
            print(f"[LunaSAM3Detector] Error in SAM3 grounding: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _select_detections(
        self,
        detections: List[Dict],
        mode: str,
        max_count: int,
        image_size: Tuple[int, int]
    ) -> List[Dict]:
        """
        Select detections based on mode.
        
        Modes:
        - largest: Sort by area, take top N
        - center: Sort by distance to image center, take top N
        - first: Take first N detections
        - all: Take all detections (up to max_count)
        """
        if mode == "largest":
            detections = sorted(
                detections,
                key=lambda d: self._bbox_area(d["bbox"]),
                reverse=True
            )
        elif mode == "center":
            img_center = (image_size[0] / 2, image_size[1] / 2)
            detections = sorted(
                detections,
                key=lambda d: self._bbox_center_distance(d["bbox"], img_center)
            )
        elif mode == "first":
            pass  # Keep original order
        elif mode == "all":
            pass  # Keep all
        
        return detections[:max_count]
    
    def _bbox_area(self, bbox: List[float]) -> float:
        """Calculate bbox area."""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)
    
    def _bbox_center_distance(self, bbox: List[float], center: Tuple[float, float]) -> float:
        """Calculate distance from bbox center to image center."""
        x1, y1, x2, y2 = bbox
        bbox_cx = (x1 + x2) / 2
        bbox_cy = (y1 + y2) / 2
        dx = bbox_cx - center[0]
        dy = bbox_cy - center[1]
        return (dx*dx + dy*dy) ** 0.5
    
    def _normalize_bbox(self, bbox: List[float], width: int, height: int) -> List[float]:
        """Normalize bbox coordinates to [0, 1] range."""
        x1, y1, x2, y2 = bbox
        return [
            x1 / width,
            y1 / height,
            x2 / width,
            y2 / height
        ]
    
    def _encode_prompts(self, clip, base_positive, detections: List[Dict]) -> List:
        """
        Encode prompts for all detections.
        
        If concept has custom prompt: encode it with CLIP
        If concept prompt empty: use base_positive conditioning
        
        This matches UltimateSDUpscale/FaceDetailer behavior.
        """
        encoded_list = []
        
        for det in detections:
            prompt_text = det.get("prompt", "").strip()
            
            if prompt_text:
                # Custom prompt - encode with CLIP
                tokens = clip.tokenize(prompt_text)
                cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
                encoded_list.append([[cond, {"pooled_output": pooled}]])
            else:
                # No custom prompt - use base positive conditioning
                encoded_list.append(base_positive)
        
        return encoded_list
    
    def _build_detection_pipe(
        self,
        detections: List[Dict],
        positive_batch: List,
        image_size: Tuple[int, int],
        padding_factor: float
    ) -> Dict:
        """
        Build final detection pipe.
        
        Structure:
        {
            "detections": [
                {
                    "concept": "face",
                    "layer": 0,
                    "bbox_norm": [x1, y1, x2, y2],
                    "mask": torch.Tensor,
                    "confidence": 0.95
                    # NO prompt string here
                },
                ...
            ],
            "positive_batch": [...],  # Pre-encoded conditioning
            "image_size": (width, height),
            "padding_factor": 1.15
        }
        """
        # Remove prompt strings from detections (already encoded)
        clean_detections = []
        for det in detections:
            clean_det = {
                "concept": det["concept"],
                "layer": det["layer"],
                "bbox_norm": det["bbox_norm"],
                "mask": det["mask"],
                "confidence": det["confidence"]
            }
            clean_detections.append(clean_det)
        
        return {
            "detections": clean_detections,
            "positive_batch": positive_batch,
            "image_size": image_size,
            "padding_factor": padding_factor
        }
    
    def _empty_detection_data(self, width: int, height: int) -> Dict:
        """Create empty detection data structure."""
        return {
            "detections": [],
            "image_size": (width, height),
            "padding_factor": 1.0
        }


# Node registration
NODE_CLASS_MAPPINGS = {
    "LunaSAM3Detector": LunaSAM3Detector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaSAM3Detector": "🌙 Luna: SAM3 Detector",
}
