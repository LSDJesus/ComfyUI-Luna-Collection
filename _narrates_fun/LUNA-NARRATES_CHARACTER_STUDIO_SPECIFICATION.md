# LUNA-Narrates: Character Creation Studio Specification

**Created:** 2025-11-11
**Version:** 1.0  
**Date:** November 11, 2025  
**Status:** Design Specification - Revenue Model

---

## Table of Contents

1. [Overview](#overview)
2. [Character Creator Workflow](#character-creator-workflow)
3. [Technical Architecture](#technical-architecture)
4. [Expression Transfer Pipeline](#expression-transfer-pipeline)
5. [Pricing & Monetization](#pricing--monetization)
6. [Database Schema](#database-schema)
7. [WebUI Design](#webui-design)
8. [Implementation Roadmap](#implementation-roadmap)

---

## Overview

### The Innovation

Instead of **text-prompting character appearance** (inconsistent, trial-and-error), users **design characters visually** like a video game character creator, then generate a **high-quality, consistent expression pack** using HyperLoRA + ControlNet/Reactor face transfer.

### Key Advantages

✅ **User Control:** Visual sliders/dropdowns (not prompt engineering)  
✅ **Consistency Guaranteed:** HyperLoRA trained on user-approved images  
✅ **Speed:** 6-8 seconds per image (768→1536 with 2x upscale + detailer)  
✅ **Parallel Processing:** Generate all 20+ expressions simultaneously  
✅ **Revenue Stream:** Charge for prototype generations + expression pack  
✅ **Quality:** Professional-grade character portraits with perfect expression transfer  

---

## Character Creator Workflow

### Phase 1: Character Design (Visual Builder)

**User Interface:** Slider/dropdown-based character customization

#### Body & Face Structure
- **Gender:** Male, Female, Non-binary, Custom
- **Age Appearance:** Child (8-12), Teen (13-17), Young Adult (18-25), Adult (26-40), Middle-Aged (41-60), Elderly (60+)
- **Ethnicity/Skin Tone:** Slider (fair → tan → deep) or preset options
- **Body Type:** Thin, Athletic, Average, Curvy, Muscular, Heavy
- **Height:** Short, Average, Tall (affects composition/framing)
- **Face Shape:** Oval, Round, Square, Heart, Diamond, Oblong

#### Facial Features
- **Eyes:**
  - Size: Small, Medium, Large, Very Large
  - Shape: Almond, Round, Hooded, Upturned, Downturned
  - Color: Brown, Blue, Green, Hazel, Gray, Amber, Heterochromia (two colors)
  - Special: Glasses, Heterochromia, Glowing, Slit Pupils
  
- **Eyebrows:**
  - Thickness: Thin, Medium, Thick
  - Shape: Straight, Arched, Angled, S-Curve
  
- **Nose:**
  - Size: Small, Medium, Large
  - Shape: Button, Straight, Roman, Upturned, Hooked
  
- **Lips:**
  - Size: Thin, Medium, Full, Very Full
  - Shape: Bow, Straight, Curved
  
- **Ears:**
  - Type: Human, Elf (pointed), Animal (cat, fox, etc.)
  - Piercings: None, Single, Multiple
  
- **Facial Hair (if applicable):**
  - Style: Clean-shaven, Stubble, Goatee, Full Beard, Mustache
  - Color: Matches hair, Different color

#### Hair
- **Length:** Bald, Buzz Cut, Short, Shoulder-Length, Long, Very Long (waist+)
- **Style:** Straight, Wavy, Curly, Braids, Ponytail, Bun, Pigtails, Mohawk, Dreadlocks
- **Color:** Black, Brown, Blonde, Red, White, Gray, Fantasy (pink, blue, etc.), Gradient/Ombre
- **Bangs:** None, Side-Swept, Straight, Curtain

#### Special Features
- **Fantasy Elements:**
  - Horns (demon, dragon, etc.)
  - Wings (angel, demon, fairy, dragon)
  - Tail (demon, cat, fox, dragon)
  - Non-human skin (scales, fur patterns)
  
- **Markings:**
  - Tattoos (location, style)
  - Scars (location, severity)
  - Birthmarks/Beauty Marks
  - Freckles
  
- **Accessories:**
  - Earrings, Necklace, Choker
  - Headband, Hair Clips
  - Piercings (nose, eyebrow, lip)

#### Clothing Style
- **Base Style:** Casual, Formal, Fantasy, Sci-Fi, Medieval, Modern, Gothic, Athletic
- **Top:** T-shirt, Tank Top, Dress Shirt, Hoodie, Armor, Robes, etc.
- **Color Palette:** Warm tones, Cool tones, Monochrome, Vibrant, Pastel, Dark
- **Accessories:** Cloak, Scarf, Jacket, Hat

#### Art Style Selection
- **Cartoon:** Western animation style (Pixar, Disney)
- **Anime:** Japanese anime style (Ghibli, shonen, shoujo)
- **Hentai:** Explicit anime style (adult content)
- **Semi-Realistic:** Painterly, stylized realism
- **Realistic:** Photorealistic 3D rendering
- **3D CG:** Stylized 3D (game engine aesthetic)

---

### Phase 2: Prototype Generation (4x Variants)

**Process:**
1. User completes character design
2. Clicks **"Generate Character"**
3. System converts selections to SDXL prompt
4. Generates **4 variant images** (same character, slight variations)
5. **768x768 base** → **2-pass detailing** → **2x upscale to 1536x1536**
6. Display 4 variants in grid

**Generation Time:** 6-8 seconds per image × 4 = **24-32 seconds total** (parallel processing)

**User Actions:**
- ✅ **Select 1-4 images** they like (checkboxes)
- 🔄 **Regenerate** for 4 new variants (if unsatisfied)
- 🎨 **Adjust parameters** and regenerate

**Iteration Limit:** 5 free regenerations, then charge per additional batch

---

### Phase 3: HyperLoRA Training

**Trigger:** User selects 1-4 approved prototype images and clicks **"Create Expression Pack"**

**Process:**
1. Take 1-4 user-approved images as training set
2. Train **character-specific HyperLoRA** (zero-shot, ~45 seconds)
3. LoRA captures facial identity, proportions, coloring
4. Save LoRA: `data/loras/character_packs/char_{uuid}_identity.safetensors`

**Quality Boost:** Multiple training images (2-4) = better LoRA consistency

---

### Phase 4: Expression Transfer (20-24 Expressions)

**Concept:** Transfer learned character face to **generic expression templates**

#### Expression Template Library

**Pre-Made Base Expressions (20-24):**
- neutral_resting
- smile_gentle
- smile_wide_happy
- smirk_confident
- smirk_mischievous
- laugh_eyes_closed
- frown_slight_confused
- frown_deep_angry
- angry_glaring_intense
- sad_downcast_eyes
- sad_crying_tears
- worried_anxious_biting_lip
- shocked_wide_eyes_mouth_open
- embarrassed_blushing_looking_away
- seductive_bedroom_eyes_sultry
- determined_clenched_jaw_fierce
- thoughtful_hand_on_chin_pondering
- suspicious_squinting_narrowed_eyes
- curious_head_tilted_inquisitive
- exhausted_tired_bags_under_eyes
- disgusted_nose_wrinkled
- excited_sparkling_eyes
- neutral_serious_stern
- playful_tongue_out_winking

**Template Images:** Generic, low-detail faces with **perfect expression capture**

---

#### Transfer Workflow (ComfyUI)

**Method 1: ControlNet Face Transfer**
```
[Expression Template Image] 
    ↓
[ControlNet Canny/OpenPose] (extract facial structure)
    ↓
[SDXL Generation with Character LoRA loaded]
    ↓
[Apply character prompt + expression tags]
    ↓
[Result: Character face with template expression]
```

**Method 2: Reactor Face Swap (Alternative/Backup)**
```
[User's approved prototype image] (source face)
    ↓
[Reactor Node: Face Detection + Extraction]
    ↓
[Expression Template Image] (target expression structure)
    ↓
[Reactor: Swap face while preserving expression]
    ↓
[Result: User's character face with template expression]
```

**Hybrid Approach (Best Quality):**
1. ControlNet extracts expression structure from template
2. SDXL generates with character LoRA
3. Reactor fine-tunes facial identity preservation
4. Two-pass detailer enhances final quality
5. 2x upscale to 1536x1536

**Per-Expression Generation:** 6-8 seconds  
**Parallel Processing:** Generate all 20-24 expressions simultaneously  
**Total Time:** ~8 seconds (not 20×8=160s, because parallel)  
**Hardware:** Batch size limited by VRAM (4-6 parallel on 24GB GPU)

---

### Phase 5: Expression Pack Delivery

**Output:**
- **20-24 high-quality expression images** (1536x1536)
- **Character HyperLoRA** (identity preservation for future custom images)
- **Character metadata JSON** (all design choices saved)
- **Expression manifest** (maps tags to images)

**Storage:**
```
data/character_packs/char_{uuid}/
  ├── identity_lora.safetensors
  ├── metadata.json
  ├── expressions/
  │   ├── neutral_resting.png
  │   ├── smile_gentle.png
  │   ├── smirk_confident.png
  │   └── ... (20-24 total)
  └── prototypes/
      ├── prototype_01_selected.png
      ├── prototype_02_selected.png
      └── ... (user-approved originals)
```

---

## Technical Architecture

### Character Design to SDXL Prompt Conversion

```python
# core/services/character_prompt_builder.py
class CharacterPromptBuilder:
    """Converts visual character design to SDXL-optimized prompt."""
    
    def build_prompt(self, character_design: CharacterDesign) -> str:
        """
        Convert character design selections to comma-delimited SDXL prompt.
        
        Args:
            character_design: User's character design choices
        
        Returns:
            SDXL-formatted positive prompt
        """
        tags = []
        
        # Gender & subject count
        if character_design.gender == "female":
            tags.append("1girl")
        elif character_design.gender == "male":
            tags.append("1boy")
        else:
            tags.append("1person")
        
        # Age appearance
        age_tags = {
            "child": "child, young",
            "teen": "teenager, youthful",
            "young_adult": "young adult, 20s",
            "adult": "adult, 30s",
            "middle_aged": "middle-aged, mature",
            "elderly": "elderly, aged"
        }
        tags.append(age_tags[character_design.age_appearance])
        
        # Ethnicity/Skin Tone
        if character_design.skin_tone:
            tags.append(f"{character_design.skin_tone} skin")
        
        # Face shape
        if character_design.face_shape:
            tags.append(f"{character_design.face_shape} face shape")
        
        # Eyes
        tags.append(f"{character_design.eye_size} eyes")
        tags.append(f"{character_design.eye_shape} eyes")
        tags.append(f"{character_design.eye_color} eyes")
        
        if character_design.eye_special:
            if "glasses" in character_design.eye_special:
                tags.append("wearing glasses")
            if "heterochromia" in character_design.eye_special:
                tags.append("heterochromia, different colored eyes")
        
        # Eyebrows
        tags.append(f"{character_design.eyebrow_thickness} eyebrows")
        
        # Nose
        tags.append(f"{character_design.nose_shape} nose")
        
        # Lips
        tags.append(f"{character_design.lip_size} lips")
        
        # Ears
        if character_design.ear_type == "elf":
            tags.append("elf ears, pointed ears")
        elif character_design.ear_type.startswith("animal"):
            animal = character_design.ear_type.split("_")[1]  # "animal_cat" -> "cat"
            tags.append(f"{animal} ears")
        
        # Facial hair
        if character_design.facial_hair and character_design.facial_hair != "clean_shaven":
            tags.append(character_design.facial_hair)
        
        # Hair
        tags.append(f"{character_design.hair_length} hair")
        tags.append(f"{character_design.hair_style} hair")
        tags.append(f"{character_design.hair_color} hair")
        
        if character_design.hair_bangs:
            tags.append(f"{character_design.hair_bangs} bangs")
        
        # Fantasy elements
        if character_design.horns:
            tags.append(f"{character_design.horns} horns")
        if character_design.wings:
            tags.append(f"{character_design.wings} wings")
        if character_design.tail:
            tags.append(f"{character_design.tail} tail")
        
        # Markings
        if character_design.tattoos:
            tags.append("tattoos")
        if character_design.scars:
            tags.append("facial scars")
        if character_design.freckles:
            tags.append("freckles")
        
        # Clothing
        tags.append(character_design.clothing_style)
        tags.append(character_design.top_garment)
        if character_design.color_palette:
            tags.append(f"{character_design.color_palette} colors")
        
        # Composition
        tags.append("portrait")
        tags.append("upper body")
        tags.append("centered")
        tags.append("detailed face")
        tags.append("high quality")
        
        # Art style keywords (from user selection)
        style_keywords = self._get_style_keywords(character_design.art_style)
        tags.extend(style_keywords)
        
        # Quality tags
        tags.extend(["masterpiece", "best quality", "detailed"])
        
        return ", ".join(tags)
    
    def _get_style_keywords(self, art_style: str) -> List[str]:
        """Map art style selection to SDXL keywords."""
        style_map = {
            "cartoon": ["western animation", "pixar style", "disney style", "cartoon"],
            "anime": ["anime", "anime style", "japanese animation"],
            "hentai": ["hentai", "anime", "explicit style"],
            "semi_realistic": ["semi-realistic", "painterly", "stylized realism"],
            "realistic": ["photorealistic", "realistic", "3d render"],
            "3d_cg": ["3d cg", "game engine", "stylized 3d", "cel shaded"]
        }
        return style_map.get(art_style, ["anime"])  # Default to anime
```

**Example Output:**
```
Input: Female, young adult, fair skin, large blue eyes, long blonde hair, elf ears, 
       leather armor, anime style

Output: "1girl, young adult, 20s, fair skin, large blue eyes, almond eyes, 
        blue eyes, medium eyebrows, button nose, medium lips, elf ears, 
        pointed ears, long hair, straight hair, blonde hair, side-swept bangs, 
        leather armor, fantasy style, warm colors, portrait, upper body, 
        centered, detailed face, high quality, anime, anime style, 
        masterpiece, best quality, detailed"
```

---

### Expression Transfer System

```python
# core/services/expression_transfer.py
class ExpressionTransferEngine:
    """Transfers character identity to expression templates."""
    
    def __init__(
        self,
        comfyui_client: ComfyUIClient,
        expression_template_path: str = "data/expression_templates/"
    ):
        self.comfyui_client = comfyui_client
        self.expression_template_path = Path(expression_template_path)
    
    async def generate_expression_pack(
        self,
        character_lora_path: str,
        character_prompt: str,
        expression_tags: List[str],
        art_style: str,
        output_dir: str
    ) -> Dict[str, str]:
        """
        Generate full expression pack using ControlNet + character LoRA.
        
        Args:
            character_lora_path: Path to trained character HyperLoRA
            character_prompt: SDXL prompt for character appearance
            expression_tags: List of expression tags to generate
            art_style: User's chosen art style
            output_dir: Where to save generated expressions
        
        Returns:
            Dict mapping expression_tag → image_path
        """
        expression_images = {}
        
        # Build ComfyUI workflow with ControlNet + LoRA
        base_workflow = self._build_expression_workflow(
            character_lora_path=character_lora_path,
            character_prompt=character_prompt,
            art_style=art_style
        )
        
        # Generate expressions in parallel batches (4-6 at a time for VRAM)
        batch_size = 4  # Adjust based on available VRAM
        
        for i in range(0, len(expression_tags), batch_size):
            batch_tags = expression_tags[i:i+batch_size]
            
            # Process batch in parallel
            batch_tasks = []
            for expr_tag in batch_tags:
                task = self._generate_single_expression(
                    workflow=base_workflow,
                    expression_tag=expr_tag,
                    output_dir=output_dir
                )
                batch_tasks.append(task)
            
            # Wait for batch completion
            batch_results = await asyncio.gather(*batch_tasks)
            
            # Map results
            for expr_tag, image_path in zip(batch_tags, batch_results):
                expression_images[expr_tag] = image_path
        
        return expression_images
    
    def _build_expression_workflow(
        self,
        character_lora_path: str,
        character_prompt: str,
        art_style: str
    ) -> Dict:
        """
        Build ComfyUI workflow JSON with ControlNet + LoRA pipeline.
        
        Workflow structure:
        1. Load expression template image
        2. ControlNet Canny/OpenPose (extract facial structure)
        3. Load SDXL checkpoint
        4. Load character LoRA
        5. Apply character prompt + expression-specific tags
        6. Generate image (768x768)
        7. Two-pass detailer (face enhancement)
        8. 2x upscale to 1536x1536
        9. Save image
        """
        workflow = {
            # Node 1: Load expression template
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": "PLACEHOLDER_TEMPLATE_PATH"}
            },
            
            # Node 2: ControlNet Preprocessor (Canny edge detection)
            "2": {
                "class_type": "CannyEdgePreprocessor",
                "inputs": {
                    "image": ["1", 0],
                    "low_threshold": 100,
                    "high_threshold": 200
                }
            },
            
            # Node 3: Load SDXL checkpoint
            "3": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": self._get_checkpoint_for_style(art_style)}
            },
            
            # Node 4: Load character LoRA
            "4": {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": ["3", 0],
                    "clip": ["3", 1],
                    "lora_name": character_lora_path,
                    "strength_model": 1.0,
                    "strength_clip": 1.0
                }
            },
            
            # Node 5: ControlNet application
            "5": {
                "class_type": "ControlNetApplyAdvanced",
                "inputs": {
                    "positive": ["6", 0],  # From CLIP encoding
                    "negative": ["7", 0],
                    "control_net": ["8", 0],  # ControlNet model
                    "image": ["2", 0],  # Preprocessed image
                    "strength": 0.85
                }
            },
            
            # Node 6: CLIP Text Encode (Positive)
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "PLACEHOLDER_POSITIVE_PROMPT",
                    "clip": ["4", 1]
                }
            },
            
            # Node 7: CLIP Text Encode (Negative)
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "blurry, low quality, worst quality, deformed, bad anatomy",
                    "clip": ["4", 1]
                }
            },
            
            # Node 8: Load ControlNet Model
            "8": {
                "class_type": "ControlNetLoader",
                "inputs": {"control_net_name": "control_v11p_sd15_canny.pth"}
            },
            
            # Node 9: KSampler (Image generation)
            "9": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["4", 0],
                    "positive": ["5", 0],
                    "negative": ["5", 1],
                    "latent_image": ["10", 0],  # Empty latent
                    "seed": "PLACEHOLDER_SEED",
                    "steps": 30,
                    "cfg": 7.0,
                    "sampler_name": "dpmpp_2m",
                    "scheduler": "karras",
                    "denoise": 1.0
                }
            },
            
            # Node 10: Empty Latent Image (768x768)
            "10": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 768, "height": 768, "batch_size": 1}
            },
            
            # Node 11: VAE Decode
            "11": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["9", 0],
                    "vae": ["3", 2]
                }
            },
            
            # Node 12: FaceDetailer (Two-pass enhancement)
            "12": {
                "class_type": "FaceDetailer",
                "inputs": {
                    "image": ["11", 0],
                    "model": ["4", 0],
                    "clip": ["4", 1],
                    "vae": ["3", 2],
                    "guide_size": 512,
                    "guide_size_for": True,
                    "max_size": 1024,
                    "seed": "PLACEHOLDER_SEED",
                    "steps": 20,
                    "cfg": 8.0,
                    "sampler_name": "dpmpp_2m",
                    "scheduler": "karras",
                    "denoise": 0.4
                }
            },
            
            # Node 13: Upscale (2x to 1536x1536)
            "13": {
                "class_type": "ImageScaleBy",
                "inputs": {
                    "image": ["12", 0],
                    "upscale_method": "lanczos",
                    "scale_by": 2.0
                }
            },
            
            # Node 14: Save Image
            "14": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["13", 0],
                    "filename_prefix": "PLACEHOLDER_FILENAME"
                }
            }
        }
        
        return workflow
    
    async def _generate_single_expression(
        self,
        workflow: Dict,
        expression_tag: str,
        output_dir: str
    ) -> str:
        """Generate single expression image."""
        
        # Load expression template
        template_path = self.expression_template_path / f"{expression_tag}.png"
        
        # Update workflow with expression-specific values
        expression_workflow = workflow.copy()
        expression_workflow["1"]["inputs"]["image"] = str(template_path)
        expression_workflow["6"]["inputs"]["text"] += f", {self._expression_to_prompt(expression_tag)}"
        expression_workflow["9"]["inputs"]["seed"] = random.randint(0, 2**32)
        expression_workflow["14"]["inputs"]["filename_prefix"] = f"char_expr_{expression_tag}"
        
        # Queue workflow in ComfyUI
        prompt_id = await self.comfyui_client.queue_prompt(expression_workflow)
        
        # Wait for completion
        image_path = await self.comfyui_client.wait_for_completion(
            prompt_id=prompt_id,
            timeout=30  # 30 seconds max per expression
        )
        
        # Move to output directory
        final_path = Path(output_dir) / f"{expression_tag}.png"
        shutil.move(image_path, final_path)
        
        return str(final_path)
    
    def _expression_to_prompt(self, expression_tag: str) -> str:
        """Convert expression tag to SDXL prompt additions."""
        expression_prompts = {
            "neutral_resting": "neutral expression, resting face, calm",
            "smile_gentle": "gentle smile, soft expression, kind eyes",
            "smile_wide_happy": "wide smile, very happy, joyful expression, big grin",
            "smirk_confident": "confident smirk, cocky expression, one-sided smile",
            "smirk_mischievous": "mischievous smirk, playful expression, scheming",
            "laugh_eyes_closed": "laughing, eyes closed, very happy, mouth open",
            "frown_slight_confused": "slight frown, confused expression, uncertain",
            "frown_deep_angry": "deep frown, angry expression, furrowed brow",
            "angry_glaring_intense": "angry glare, intense expression, narrowed eyes, hostile",
            "sad_downcast_eyes": "sad expression, downcast eyes, looking down, melancholy",
            "sad_crying_tears": "crying, tears streaming, very sad, emotional",
            "worried_anxious_biting_lip": "worried expression, anxious, biting lip, concerned",
            "shocked_wide_eyes_mouth_open": "shocked expression, wide eyes, mouth open, surprised",
            "embarrassed_blushing_looking_away": "embarrassed, blushing, red cheeks, looking away, shy",
            "seductive_bedroom_eyes_sultry": "seductive expression, bedroom eyes, sultry, half-lidded eyes",
            "determined_clenched_jaw_fierce": "determined expression, clenched jaw, fierce, resolute",
            "thoughtful_hand_on_chin_pondering": "thoughtful expression, hand on chin, pondering, contemplative",
            "suspicious_squinting_narrowed_eyes": "suspicious expression, squinting, narrowed eyes, distrustful",
            "curious_head_tilted_inquisitive": "curious expression, head tilted, inquisitive, interested",
            "exhausted_tired_bags_under_eyes": "exhausted expression, tired, bags under eyes, weary",
            "disgusted_nose_wrinkled": "disgusted expression, nose wrinkled, repulsed",
            "excited_sparkling_eyes": "excited expression, sparkling eyes, enthusiastic",
            "neutral_serious_stern": "serious expression, stern, stoic, neutral",
            "playful_tongue_out_winking": "playful expression, tongue out, winking, cheeky"
        }
        return expression_prompts.get(expression_tag, "neutral expression")
    
    def _get_checkpoint_for_style(self, art_style: str) -> str:
        """Map art style to SDXL checkpoint."""
        checkpoint_map = {
            "cartoon": "pixar_diffusion_xl.safetensors",
            "anime": "animagine_xl_v3.safetensors",
            "hentai": "pony_diffusion_xl_v6.safetensors",
            "semi_realistic": "juggernaut_xl_v9.safetensors",
            "realistic": "realistic_vision_xl_v5.safetensors",
            "3d_cg": "3d_animation_xl_v2.safetensors"
        }
        return checkpoint_map.get(art_style, "animagine_xl_v3.safetensors")
```

---

## Pricing & Monetization

### Revenue Model: Premium Character Packs

#### Tier 1: Character Design + Prototypes (FREE or LOW COST)
- **Included:**
  - Visual character designer access
  - First 4 prototype generations (free)
  - 5 regeneration attempts (20 total images free)
- **Cost to User:** FREE (acquisition funnel)
- **Cost to Platform:** ~$0.02 compute (20 images × $0.001/image)

**Strategy:** Get users invested in their character design before monetization

---

#### Tier 2: Standard Expression Pack ($4.99-$9.99)
- **Included:**
  - 20 core expressions (1536x1536 high quality)
  - Character HyperLoRA (.safetensors file)
  - Expression manifest JSON
  - Lifetime storage
  - Use in unlimited stories
- **Recommended Price:** $7.99
- **Cost to Platform:**
  - HyperLoRA training: $0 (local)
  - 20 expressions × 8 seconds = ~3 minutes compute
  - Compute cost: ~$0.05 (20 images × $0.0025/image for upscaled)
- **Profit Margin:** $7.94 profit per pack (99.4% margin)

---

#### Tier 3: Premium Expression Pack ($14.99-$19.99)
- **Included:**
  - 24 expressions (adds: disgusted, excited, serious, playful)
  - All Standard features
  - Priority generation queue
  - Custom expression requests (user can request specific emotions)
  - 4K upscale option (3072x3072)
- **Recommended Price:** $17.99
- **Cost to Platform:** ~$0.10 (24 expressions + 4K upscale)
- **Profit Margin:** $17.89 profit per pack (99.4% margin)

---

#### Tier 4: Ultimate Character Studio ($29.99-$49.99)
- **Included:**
  - 24 premium expressions
  - **Multiple outfit variants** (casual, formal, armor, etc.) - 3-5 outfits
  - Each outfit: 10-15 key expressions = 50-75 total images
  - Character LoRA for each outfit variant
  - **Pose variations** (sitting, standing, action poses)
  - Custom background options
  - Commercial use license
- **Recommended Price:** $39.99
- **Cost to Platform:** ~$0.50 (75 images + multiple LoRAs)
- **Profit Margin:** $39.49 profit per pack (98.8% margin)

---

### Additional Revenue Streams

#### À La Carte Options
- **Single Expression Generation:** $0.49 each
- **Custom Expression Request:** $1.99 (user describes, we generate)
- **Outfit Change Pack:** $4.99 (10 expressions in new outfit)
- **Background Variants:** $2.99 (same expression, 5 different backgrounds)
- **4K Upscale Upgrade:** $1.99 per expression

#### Subscription Model (Optional)
- **Creator Pro:** $14.99/month
  - 3 character packs per month (Standard tier)
  - Unlimited prototype regenerations
  - Priority generation queue
  - Early access to new expressions/styles
  
- **Studio Unlimited:** $29.99/month
  - 5 Premium packs per month
  - Commercial use license
  - API access for batch generation
  - Custom model fine-tuning

---

### Competitive Pricing Analysis

**Market Comparison:**
- **Artbreeder:** Free basic, $8.99/month Pro (limited generations)
- **NovelAI Image Gen:** $10/month for 1000 generations (no character consistency)
- **Character.AI:** Free but no image generation
- **Midjourney:** $10/month (no character consistency guaranteed)

**LUNA Advantage:** 
- One-time purchase (not subscription)
- **Guaranteed consistency** (HyperLoRA + ControlNet)
- **Immediate usability** (works in stories instantly)
- **High-quality expressions** (1536x1536, professionally composed)

---

## Database Schema

```sql
-- Character packs created by users
CREATE TABLE IF NOT EXISTS luna.character_packs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,  -- User who created character
    character_name TEXT NOT NULL,
    
    -- Design Data
    character_design JSONB NOT NULL,  -- All design choices saved
    art_style TEXT NOT NULL,  -- "anime", "realistic", etc.
    
    -- Generated Assets
    lora_path TEXT NOT NULL,  -- Character identity LoRA
    prototype_images JSONB NOT NULL,  -- User-selected prototype images
    expression_manifest JSONB NOT NULL,  -- Maps expression_tag → image_path
    
    -- Pricing Tier
    pack_tier TEXT NOT NULL CHECK (pack_tier IN ('free', 'standard', 'premium', 'ultimate')),
    pack_price DECIMAL(10, 2),  -- Amount charged
    
    -- Metadata
    expression_count INT DEFAULT 20,
    resolution TEXT DEFAULT '1536x1536',
    storage_path TEXT NOT NULL,  -- Base path to character_packs/char_{uuid}/
    
    -- Usage Tracking
    used_in_stories INT DEFAULT 0,  -- How many stories use this character
    total_turns_generated INT DEFAULT 0,  -- Total turns featuring this character
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(user_id, character_name)
);

CREATE INDEX idx_character_packs_user ON luna.character_packs(user_id);
CREATE INDEX idx_character_packs_tier ON luna.character_packs(pack_tier);

-- Track character pack purchases
CREATE TABLE IF NOT EXISTS luna.character_pack_purchases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    character_pack_id UUID NOT NULL REFERENCES luna.character_packs(id) ON DELETE CASCADE,
    
    -- Transaction Details
    pack_tier TEXT NOT NULL,
    amount_paid DECIMAL(10, 2) NOT NULL,
    currency TEXT DEFAULT 'USD',
    payment_method TEXT,  -- "stripe", "paypal", etc.
    transaction_id TEXT,  -- External payment processor ID
    
    -- Status
    status TEXT DEFAULT 'completed' CHECK (status IN ('pending', 'completed', 'failed', 'refunded')),
    
    -- Generation Metrics
    prototype_generation_count INT DEFAULT 4,  -- How many prototypes generated
    expression_generation_time_seconds INT,  -- Total generation time
    total_images_generated INT,  -- Total including prototypes + expressions
    
    purchased_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_pack_purchases_user ON luna.character_pack_purchases(user_id);
CREATE INDEX idx_pack_purchases_pack ON luna.character_pack_purchases(character_pack_id);
CREATE INDEX idx_pack_purchases_date ON luna.character_pack_purchases(purchased_at);

-- Link characters in stories to character packs
ALTER TABLE luna.character_matrices ADD COLUMN IF NOT EXISTS character_pack_id UUID 
    REFERENCES luna.character_packs(id) ON DELETE SET NULL;
ALTER TABLE luna.character_matrices ADD COLUMN IF NOT EXISTS uses_expression_pack BOOLEAN DEFAULT FALSE;

CREATE INDEX idx_character_matrices_pack ON luna.character_matrices(character_pack_id);
```

---

## WebUI Design

### Character Creator Page

```typescript
// webui/pages/CharacterCreator.tsx
import React, { useState } from 'react';

interface CharacterDesign {
  // Body & Face
  gender: 'male' | 'female' | 'nonbinary' | 'custom';
  ageAppearance: 'child' | 'teen' | 'young_adult' | 'adult' | 'middle_aged' | 'elderly';
  skinTone: string;
  bodyType: string;
  height: string;
  faceShape: string;
  
  // Eyes
  eyeSize: string;
  eyeShape: string;
  eyeColor: string;
  eyeSpecial: string[];
  
  // Other facial features...
  
  // Hair
  hairLength: string;
  hairStyle: string;
  hairColor: string;
  hairBangs: string;
  
  // Fantasy elements
  horns?: string;
  wings?: string;
  tail?: string;
  
  // Clothing
  clothingStyle: string;
  topGarment: string;
  colorPalette: string;
  
  // Art style
  artStyle: 'cartoon' | 'anime' | 'hentai' | 'semi_realistic' | 'realistic' | '3d_cg';
}

export default function CharacterCreator() {
  const [design, setDesign] = useState<CharacterDesign>({} as CharacterDesign);
  const [prototypeImages, setPrototypeImages] = useState<string[]>([]);
  const [selectedPrototypes, setSelectedPrototypes] = useState<Set<number>>(new Set());
  const [generationStage, setGenerationStage] = useState<'design' | 'prototypes' | 'expressions' | 'complete'>('design');
  
  const handleGeneratePrototypes = async () => {
    setGenerationStage('prototypes');
    
    const response = await fetch('/api/character-packs/generate-prototypes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ characterDesign: design })
    });
    
    const data = await response.json();
    setPrototypeImages(data.imageUrls);
  };
  
  const handleCreateExpressionPack = async (tier: string) => {
    setGenerationStage('expressions');
    
    const selectedImages = Array.from(selectedPrototypes).map(i => prototypeImages[i]);
    
    const response = await fetch('/api/character-packs/create-pack', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        characterDesign: design,
        selectedPrototypes: selectedImages,
        packTier: tier
      })
    });
    
    const data = await response.json();
    
    // Poll for completion
    const taskId = data.taskId;
    pollExpressionGeneration(taskId);
  };
  
  return (
    <div className="character-creator-container">
      {generationStage === 'design' && (
        <DesignStage 
          design={design}
          setDesign={setDesign}
          onGenerate={handleGeneratePrototypes}
        />
      )}
      
      {generationStage === 'prototypes' && (
        <PrototypeStage
          images={prototypeImages}
          selectedPrototypes={selectedPrototypes}
          setSelectedPrototypes={setSelectedPrototypes}
          onCreatePack={handleCreateExpressionPack}
          onRegenerate={handleGeneratePrototypes}
        />
      )}
      
      {generationStage === 'expressions' && (
        <ExpressionGenerationProgress />
      )}
      
      {generationStage === 'complete' && (
        <ExpressionPackComplete />
      )}
    </div>
  );
}

function DesignStage({ design, setDesign, onGenerate }: any) {
  return (
    <div className="design-stage">
      <h1>Create Your Character</h1>
      
      <div className="design-grid">
        {/* Left: Design controls */}
        <div className="design-controls">
          <DesignSection title="Body & Face">
            <Select label="Gender" value={design.gender} onChange={...} options={...} />
            <Select label="Age" value={design.ageAppearance} onChange={...} />
            <Slider label="Skin Tone" value={design.skinTone} onChange={...} />
            {/* More controls... */}
          </DesignSection>
          
          <DesignSection title="Eyes">
            <Select label="Size" value={design.eyeSize} options={['small', 'medium', 'large', 'very_large']} />
            <Select label="Shape" value={design.eyeShape} options={['almond', 'round', 'hooded']} />
            <ColorPicker label="Color" value={design.eyeColor} />
            {/* More controls... */}
          </DesignSection>
          
          {/* More sections: Hair, Clothing, Fantasy Elements, etc. */}
        </div>
        
        {/* Right: Live preview (optional) */}
        <div className="design-preview">
          <CharacterPreview design={design} />
          <button onClick={onGenerate} className="generate-btn">
            Generate Character Prototypes
          </button>
        </div>
      </div>
    </div>
  );
}

function PrototypeStage({ 
  images, 
  selectedPrototypes, 
  setSelectedPrototypes, 
  onCreatePack,
  onRegenerate 
}: any) {
  return (
    <div className="prototype-stage">
      <h1>Select Your Character</h1>
      <p>Choose 1-4 images that match your vision, or regenerate for new options</p>
      
      <div className="prototype-grid">
        {images.map((url, index) => (
          <div 
            key={index}
            className={`prototype-card ${selectedPrototypes.has(index) ? 'selected' : ''}`}
            onClick={() => {
              const newSelection = new Set(selectedPrototypes);
              if (newSelection.has(index)) {
                newSelection.delete(index);
              } else {
                newSelection.add(index);
              }
              setSelectedPrototypes(newSelection);
            }}
          >
            <img src={url} alt={`Prototype ${index + 1}`} />
            {selectedPrototypes.has(index) && (
              <div className="selection-indicator">✓</div>
            )}
          </div>
        ))}
      </div>
      
      <div className="action-buttons">
        <button onClick={onRegenerate} className="regenerate-btn">
          🔄 Regenerate (5 free remaining)
        </button>
        
        <button 
          onClick={() => onCreatePack('standard')} 
          disabled={selectedPrototypes.size === 0}
          className="create-pack-btn"
        >
          Create Standard Pack ($7.99) - 20 Expressions
        </button>
        
        <button 
          onClick={() => onCreatePack('premium')} 
          disabled={selectedPrototypes.size === 0}
          className="create-pack-btn premium"
        >
          Create Premium Pack ($17.99) - 24 Expressions
        </button>
      </div>
    </div>
  );
}
```

---

## Implementation Roadmap

### Phase 1: Expression Template Library (Week 1)

**Goal:** Create high-quality expression templates

**Tasks:**
1. Generate 24 generic expression templates (neutral face, perfect expressions)
2. Store in `data/expression_templates/`
3. Document expression tag → template mapping
4. Test ControlNet edge detection on templates

**Deliverables:**
- 24 expression template images (768x768 base)
- Expression manifest JSON

---

### Phase 2: Character Design System (Week 1-2)

**Goal:** Build visual character creator

**Tasks:**
1. Create `CharacterDesign` Pydantic model
2. Implement `CharacterPromptBuilder` service
3. Build WebUI design interface (sliders, dropdowns, color pickers)
4. Test prompt quality with sample generations

**Deliverables:**
- Character design data model
- Prompt builder with 50+ design parameters
- Basic WebUI for character design

---

### Phase 3: Prototype Generation (Week 2)

**Goal:** Generate 4 prototype variants

**Tasks:**
1. Implement prototype generation endpoint
2. Add SDXL generation with art style checkpoints
3. Add 768→1536 upscale pipeline
4. Build prototype selection UI

**Deliverables:**
- `/api/character-packs/generate-prototypes` endpoint
- Prototype generation in 24-32 seconds
- Selection UI with checkboxes

---

### Phase 4: Expression Transfer Pipeline (Week 2-3)

**Goal:** Implement ControlNet + LoRA workflow

**Tasks:**
1. Create `ExpressionTransferEngine` service
2. Build ComfyUI workflow with ControlNet + LoRA
3. Implement parallel batch processing (4-6 expressions simultaneously)
4. Test expression quality and consistency

**Deliverables:**
- Expression transfer pipeline
- 20 expressions generated in ~2-3 minutes (parallel)
- Quality validation

---

### Phase 5: Payment Integration (Week 3)

**Goal:** Implement tier-based purchasing

**Tasks:**
1. Add `character_packs` and `character_pack_purchases` tables
2. Integrate Stripe for payments
3. Build pricing tier selection UI
4. Implement pack download/delivery

**Deliverables:**
- Payment processing
- Pack tier selection
- Secure asset delivery

---

### Phase 6: Story Integration (Week 3-4)

**Goal:** Use character packs in stories

**Tasks:**
1. Link `character_matrices` to `character_packs`
2. Auto-load expression pack in conversational mode
3. Expression matching uses pack expressions
4. Track usage analytics

**Deliverables:**
- Character pack integration in stories
- Expression bank auto-populated from pack
- Usage tracking

---

### Phase 7: Polish & Launch (Week 4)

**Goal:** Production-ready character studio

**Tasks:**
1. Add generation progress tracking
2. Build expression gallery view
3. Implement custom expression requests
4. Add pack sharing/marketplace (future)
5. Launch marketing campaign

**Deliverables:**
- Production character creator
- Marketing materials
- User documentation

---

## Success Metrics

### Target Metrics (6 Months)

**User Adoption:**
- 10,000 character packs created (free prototypes)
- 2,500 paid packs purchased (25% conversion)
- $20,000 revenue from character packs

**Average Revenue Per User (ARPU):**
- Conversion rate: 25% (free prototype → paid pack)
- Average pack price: $8 (Standard tier most popular)
- ARPU: $2 per user

**Technical Performance:**
- Prototype generation: <30 seconds
- Expression pack generation: <5 minutes (20 expressions)
- 99.5% success rate (generation completion)
- 4.8/5 user satisfaction (character quality)

---

---

## Advanced Feature: Full-Body Character Studio (Phase 2 Expansion)

### Vision: 3D-Powered Character Generation

**Concept:** Expand beyond facial expressions to **full-body character creation** with:
- Interactive 3D mesh model for pose control
- Preset poses + custom posing capability
- Controlnet generation from 3D mesh (pose + depth)
- Img2img workflows for outfit/clothing customization
- HyperLoRA + slider LoRAs for body consistency

**Technical Stack:**
- **3D Mesh Library:** Three.js + Mixamo-style rigged model
- **Pose Control:** Interactive 3D viewport with bone/joint manipulation
- **2D Capture:** Render 3D pose to 2D silhouette/depth map
- **ControlNets:** OpenPose (skeleton), Depth (spatial), Canny (edges)
- **Img2Img Pipeline:** Outfit variations via iterative refinement
- **Custom ComfyUI Nodes:** Streamlined workflow with minimal nodes

---

### Full-Body Character Workflow

#### Stage 1: Body Design (Expanded Character Creator)

**New Body Customization Options:**

**Body Proportions:**
- **Height:** 4'0" - 7'0" (120cm - 213cm) slider
- **Build:** Ultra-thin, Thin, Athletic, Average, Curvy, Muscular, Heavy, Plus-size
- **Shoulder Width:** Narrow, Average, Broad
- **Waist:** Narrow, Average, Wide
- **Hips:** Narrow, Average, Wide
- **Chest/Bust:** Flat, Small, Medium, Large, Very Large
- **Leg Length:** Short, Average, Long (affects proportions)
- **Arm Length:** Short, Average, Long

**Body Features:**
- **Musculature:** None, Toned, Athletic, Bodybuilder
- **Body Hair:** None, Light, Medium, Heavy
- **Skin Details:** Smooth, Freckled, Moles, Birthmarks, Scars
- **Tattoos:** None, Small, Medium, Large, Full Body (with placement selector)
- **Fantasy Elements:**
  - Wings (back-mounted, size slider)
  - Tail (type, length)
  - Non-human skin (scales, fur, patterns)
  - Extra limbs (4 arms, etc. for fantasy races)

**Clothing/Outfit Selection:**
- **Base Outfit Style:** Casual, Formal, Fantasy, Sci-Fi, Medieval, Athletic, Swimwear, Lingerie, Armor
- **Top:** 50+ options (T-shirt, tank, dress shirt, hoodie, corset, armor, etc.)
- **Bottom:** 50+ options (Jeans, skirt, dress, armor pants, robes, etc.)
- **Footwear:** Sneakers, boots, heels, sandals, barefoot
- **Accessories:** Cloak, scarf, belt, gloves, jewelry
- **Color Customization:** Per-item color picker
- **Material:** Leather, cloth, metal, latex, silk, etc.

---

#### Stage 2: Pose Selection/Creation

**Method A: Preset Pose Library (Easy)**

**Pose Categories:**
- **Portrait Poses:** Standing neutral, confident stance, relaxed, arms crossed
- **Action Poses:** Running, jumping, fighting stance, casting spell, drawing weapon
- **Sitting Poses:** Chair sitting, ground sitting, lotus position, sprawled
- **Lying Poses:** Back, side, stomach, curled up
- **Interaction Poses:** Reaching out, pointing, waving, beckoning, leaning against wall
- **Emotional Poses:** Confident (hands on hips), shy (arms behind back), defensive (arms crossed), seductive (hip thrust)
- **Fantasy/Combat:** Sword swing, bow draw, spellcasting, shield block, victory pose

**User Action:** Click preset pose thumbnail → Apply to 3D model

---

**Method B: Interactive 3D Pose Editor (Advanced)**

**3D Viewport Interface:**
```typescript
// webui/components/PoseEditor3D.tsx
import { Canvas } from '@react-three/fiber';
import { OrbitControls, useGLTF } from '@react-three/drei';

function PoseEditor3D({ onPoseCapture }: { onPoseCapture: (poseData: PoseData) => void }) {
  const [model, setModel] = useState<any>(null);
  const [selectedBone, setSelectedBone] = useState<string | null>(null);
  
  // Load rigged 3D model (generic humanoid)
  const { scene } = useGLTF('/models/generic_humanoid_rigged.glb');
  
  return (
    <div className="pose-editor">
      <Canvas camera={{ position: [0, 1.5, 3], fov: 50 }}>
        <ambientLight intensity={0.5} />
        <spotLight position={[10, 10, 10]} angle={0.15} />
        
        <InteractiveModel 
          scene={scene}
          selectedBone={selectedBone}
          onBoneSelect={setSelectedBone}
        />
        
        <OrbitControls />
      </Canvas>
      
      <BoneControls 
        selectedBone={selectedBone}
        onRotate={(axis, angle) => rotateBone(selectedBone, axis, angle)}
      />
      
      <button onClick={() => capturePose(model)}>
        Capture Pose
      </button>
    </div>
  );
}
```

**Pose Control Features:**
- Click bone/joint to select (highlight in 3D view)
- Rotation sliders (X, Y, Z axes) or direct 3D manipulation
- IK (Inverse Kinematics) for natural limb positioning
- Symmetry mode (mirror left → right for balanced poses)
- Preset bone chains (arm, leg, spine) for quick adjustments
- Reset to T-pose/A-pose
- Save custom poses to library

**2D Capture Process:**
1. User finishes posing 3D model
2. Click "Capture Pose"
3. System renders multiple outputs:
   - **RGB Silhouette** (solid color body shape)
   - **Depth Map** (grayscale depth information)
   - **OpenPose Skeleton** (stick figure joint map)
   - **Canny Edge Map** (edge detection)
4. Store all 4 maps as ControlNet inputs

---

#### Stage 3: Character Generation (Full-Body)

**ComfyUI Workflow: 3D Pose → AI Character**

```python
# Workflow Structure (Custom ComfyUI Node Implementation)

class LUNACharacterGeneratorNode:
    """
    Custom ComfyUI node for full-body character generation.
    
    Inputs:
    - character_design_json: Character design parameters
    - pose_image: 2D render from 3D model (RGB silhouette)
    - depth_map: Depth information from 3D model
    - openpose_map: Skeleton joint map
    - character_lora: Character-specific HyperLoRA
    - outfit_prompt: Clothing/outfit description
    - art_style: User's chosen art style
    
    Outputs:
    - generated_character: Full-body character image (1024x1536 portrait or 1536x1024 landscape)
    """
    
    def __init__(self):
        self.controlnet_models = {
            'openpose': 'control_v11p_sd15_openpose.pth',
            'depth': 'control_v11f1p_sd15_depth.pth',
            'canny': 'control_v11p_sd15_canny.pth'
        }
    
    def generate(
        self,
        character_design: Dict,
        pose_image: Image,
        depth_map: Image,
        openpose_map: Image,
        character_lora_path: str,
        outfit_prompt: str,
        art_style: str
    ) -> Image:
        """
        Single-node workflow combining all inputs.
        
        Workflow:
        1. Load SDXL checkpoint (based on art_style)
        2. Load character HyperLoRA
        3. Load 3 ControlNets (OpenPose, Depth, Canny)
        4. Build prompt from character_design + outfit_prompt
        5. Apply multi-ControlNet conditioning
        6. Generate base image (1024x1536)
        7. DetailerPro pass (face + hands + feet)
        8. Optional 1.5x upscale to 1536x2304 (full quality)
        9. Return final image
        """
        
        # Build positive prompt
        prompt = self._build_full_body_prompt(character_design, outfit_prompt)
        
        # Multi-ControlNet application
        controlnet_conditioning = self._apply_multi_controlnet(
            openpose_image=openpose_map,
            depth_image=depth_map,
            canny_image=self._extract_canny(pose_image),
            weights=[0.8, 0.6, 0.4]  # OpenPose strongest, Depth medium, Canny subtle
        )
        
        # Generate with all conditioning
        generated_image = self._sdxl_generate(
            prompt=prompt,
            negative_prompt=self._get_negative_prompt(),
            controlnet_conditioning=controlnet_conditioning,
            lora_path=character_lora_path,
            width=1024,
            height=1536,
            steps=35,
            cfg=7.5
        )
        
        # Enhance details (face, hands, feet - common problem areas)
        enhanced_image = self._detailer_pass(
            image=generated_image,
            focus_regions=['face', 'hands', 'feet'],
            denoise=0.4
        )
        
        return enhanced_image
    
    def _build_full_body_prompt(self, character_design: Dict, outfit_prompt: str) -> str:
        """Combine character features + outfit into SDXL prompt."""
        
        # Character features (from CharacterPromptBuilder)
        char_tags = CharacterPromptBuilder().build_prompt(character_design)
        
        # Body-specific additions
        body_tags = []
        body_tags.append(f"{character_design['build']} build")
        body_tags.append(f"{character_design['height']}cm tall")
        
        if character_design.get('musculature') and character_design['musculature'] != 'none':
            body_tags.append(f"{character_design['musculature']} physique")
        
        # Outfit/clothing
        outfit_tags = outfit_prompt.split(',')  # Assume comma-delimited
        
        # Composition
        composition_tags = [
            "full body",
            "standing",
            "entire body visible",
            "head to toe",
            "full shot",
            "detailed anatomy",
            "well-proportioned",
            "masterpiece",
            "best quality"
        ]
        
        all_tags = char_tags.split(', ') + body_tags + outfit_tags + composition_tags
        return ', '.join(all_tags)
    
    def _apply_multi_controlnet(
        self,
        openpose_image: Image,
        depth_image: Image,
        canny_image: Image,
        weights: List[float]
    ) -> Any:
        """Apply 3 ControlNets simultaneously with different weights."""
        
        # OpenPose (strongest - controls pose/body structure)
        openpose_conditioning = self._apply_controlnet(
            controlnet_type='openpose',
            image=openpose_image,
            strength=weights[0]
        )
        
        # Depth (medium - controls spatial depth)
        depth_conditioning = self._apply_controlnet(
            controlnet_type='depth',
            image=depth_image,
            strength=weights[1]
        )
        
        # Canny (subtle - edge guidance)
        canny_conditioning = self._apply_controlnet(
            controlnet_type='canny',
            image=canny_image,
            strength=weights[2]
        )
        
        # Combine all three
        return self._combine_conditionings([
            openpose_conditioning,
            depth_conditioning,
            canny_conditioning
        ])
```

---

#### Stage 4: Outfit Variations (Img2Img)

**Goal:** Generate same character, same pose, different outfits

**Workflow:**
1. User generates base character in outfit A
2. User clicks "Change Outfit"
3. Select new outfit from presets or describe custom outfit
4. Img2Img workflow:
   - Take generated character as input image
   - **Denoise strength: 0.3-0.5** (keep pose/body, change clothing)
   - New prompt: same character features + new outfit description
   - ControlNets stay applied (preserve pose exactly)
   - Character LoRA loaded (preserve identity)
5. Result: Same character, same pose, new outfit

**Outfit Quick-Change Presets:**
- Casual (jeans + t-shirt)
- Formal (suit/dress)
- Fantasy (armor, robes, wizard outfit)
- Sci-Fi (tech suit, cyberpunk gear)
- Medieval (knight armor, peasant clothes)
- Athletic (workout clothes, sports uniform)
- Swimwear (bikini, swim trunks)
- Lingerie/Intimate
- Custom (user-described)

**Speed:** 8-12 seconds per outfit variation (Img2Img faster than text2img)

---

### Character Pack Tiers (Expanded)

#### Tier 1: Expression Pack (Facial Only) - $7.99
- 20 facial expressions (portrait, 1536x1536)
- Character HyperLoRA
- No full-body images

#### Tier 2: Portrait+ Pack - $14.99
- 20 facial expressions (1536x1536)
- 10 upper-body poses (portrait, 1024x1536)
- 1 outfit included
- Character HyperLoRA

#### Tier 3: Full-Body Standard - $24.99
- 20 facial expressions (1536x1536)
- 15 full-body poses (10 preset poses + 5 custom)
- 3 outfit variations
- Character HyperLoRA + body LoRA sliders
- 1024x1536 resolution

#### Tier 4: Full-Body Premium - $39.99
- 24 facial expressions
- 25 full-body poses (15 preset + 10 custom)
- 5 outfit variations
- Character HyperLoRA + body LoRA sliders
- 1536x2304 resolution (higher quality)
- Priority generation queue

#### Tier 5: Ultimate Character Studio - $69.99
- 30 facial expressions
- 40 full-body poses (20 preset + 20 custom)
- 10 outfit variations
- Multiple art style versions (anime + realistic)
- Character HyperLoRA + body LoRA sliders
- 1536x2304 resolution
- Custom pose editor access (save unlimited poses)
- Commercial use license

---

### Custom ComfyUI Node Architecture

**Node Structure:** Minimize workflow complexity with single-purpose nodes

#### Node 1: LUNA_PoseToControlNets
```python
class LUNA_PoseToControlNets:
    """
    Input: 3D model pose data (JSON) or 2D rendered image
    Output: OpenPose map, Depth map, Canny map (all ready for ControlNet)
    
    Handles:
    - 3D pose JSON → 2D projection
    - RGB silhouette → multi-map generation
    - Automatic preprocessing (no manual steps)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_source": (["3d_json", "2d_image"],),
                "pose_data": ("STRING", {"multiline": True}),  # JSON or base64 image
                "image_width": ("INT", {"default": 1024}),
                "image_height": ("INT", {"default": 1536})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("openpose_map", "depth_map", "canny_map")
    FUNCTION = "generate_controlnet_maps"
    
    def generate_controlnet_maps(self, pose_source, pose_data, image_width, image_height):
        if pose_source == "3d_json":
            # Parse 3D pose JSON, render to 2D
            rendered_image = self.render_3d_pose_to_2d(pose_data, image_width, image_height)
        else:
            # Decode base64 image
            rendered_image = self.decode_image(pose_data)
        
        # Generate all 3 maps
        openpose_map = self.extract_openpose(rendered_image)
        depth_map = self.extract_depth(rendered_image)
        canny_map = self.extract_canny(rendered_image)
        
        return (openpose_map, depth_map, canny_map)
```

#### Node 2: LUNA_CharacterGenerator
```python
class LUNA_CharacterGenerator:
    """
    All-in-one character generation node.
    
    Input: Character design JSON, ControlNet maps, LoRA, outfit prompt
    Output: Final character image (with detailing, upscaling)
    
    Handles:
    - Prompt building from JSON
    - Multi-ControlNet application
    - LoRA loading
    - SDXL generation
    - Automatic detailing (face, hands, feet)
    - Optional upscaling
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_design_json": ("STRING", {"multiline": True}),
                "openpose_map": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "canny_map": ("IMAGE",),
                "character_lora_path": ("STRING",),
                "outfit_prompt": ("STRING",),
                "art_style": (["anime", "realistic", "cartoon", "3d_cg"],),
                "apply_detailing": ("BOOLEAN", {"default": True}),
                "upscale_to_high_res": ("BOOLEAN", {"default": False})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("character_image",)
    FUNCTION = "generate_character"
    
    def generate_character(self, character_design_json, openpose_map, depth_map, 
                          canny_map, character_lora_path, outfit_prompt, art_style,
                          apply_detailing, upscale_to_high_res):
        
        # Parse character design
        design = json.loads(character_design_json)
        
        # Build prompt
        prompt = self.build_full_body_prompt(design, outfit_prompt, art_style)
        
        # Load models
        checkpoint = self.get_checkpoint_for_style(art_style)
        model, clip, vae = self.load_checkpoint(checkpoint)
        model = self.load_lora(model, clip, character_lora_path)
        
        # Apply ControlNets
        positive_cond = self.encode_prompt(prompt, clip)
        negative_cond = self.encode_prompt(self.get_negative_prompt(), clip)
        
        positive_cond = self.apply_multi_controlnet(
            positive_cond, negative_cond,
            openpose_map, depth_map, canny_map
        )
        
        # Generate
        latent = self.generate_empty_latent(1024, 1536)
        samples = self.ksampler(model, positive_cond, negative_cond, latent)
        image = self.vae_decode(samples, vae)
        
        # Detailing pass
        if apply_detailing:
            image = self.detail_face_hands_feet(image, model, clip, vae)
        
        # Upscale
        if upscale_to_high_res:
            image = self.upscale_image(image, scale=1.5)
        
        return (image,)
```

#### Node 3: LUNA_OutfitChanger
```python
class LUNA_OutfitChanger:
    """
    Img2Img node for quick outfit changes.
    
    Input: Generated character image, new outfit prompt
    Output: Same character, same pose, new outfit
    
    Handles:
    - Low denoise img2img (preserve body/pose)
    - ControlNet reapplication
    - LoRA consistency
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_character_image": ("IMAGE",),
                "new_outfit_prompt": ("STRING",),
                "character_lora_path": ("STRING",),
                "denoise_strength": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 0.7}),
                "preserve_pose": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("new_outfit_image",)
    FUNCTION = "change_outfit"
    
    def change_outfit(self, base_character_image, new_outfit_prompt, 
                     character_lora_path, denoise_strength, preserve_pose):
        
        # Extract ControlNets from base image (if preserve_pose)
        if preserve_pose:
            openpose_map = self.extract_openpose_from_image(base_character_image)
            depth_map = self.extract_depth_from_image(base_character_image)
        
        # Build new prompt (character features + new outfit)
        # Note: Extract character features from original generation metadata
        prompt = self.build_outfit_change_prompt(new_outfit_prompt)
        
        # Img2Img with low denoise
        new_image = self.img2img(
            image=base_character_image,
            prompt=prompt,
            lora_path=character_lora_path,
            denoise=denoise_strength,
            controlnets=(openpose_map, depth_map) if preserve_pose else None
        )
        
        return (new_image,)
```

---

### Database Schema Extensions

```sql
-- Add full-body support to character packs
ALTER TABLE luna.character_packs 
    ADD COLUMN includes_full_body BOOLEAN DEFAULT FALSE,
    ADD COLUMN pose_count INT DEFAULT 0,
    ADD COLUMN outfit_count INT DEFAULT 1,
    ADD COLUMN max_resolution TEXT DEFAULT '1536x1536',
    ADD COLUMN has_custom_pose_editor BOOLEAN DEFAULT FALSE;

-- Store custom poses created by users
CREATE TABLE IF NOT EXISTS luna.custom_poses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    character_pack_id UUID REFERENCES luna.character_packs(id) ON DELETE CASCADE,
    
    pose_name TEXT NOT NULL,
    pose_category TEXT,  -- "action", "sitting", "portrait", etc.
    
    -- 3D Pose Data
    pose_json JSONB NOT NULL,  -- Full 3D bone/joint data
    
    -- Generated ControlNet Maps
    openpose_map_url TEXT,
    depth_map_url TEXT,
    canny_map_url TEXT,
    thumbnail_url TEXT,  -- Preview of pose
    
    -- Usage
    is_public BOOLEAN DEFAULT FALSE,  -- Allow other users to use this pose
    usage_count INT DEFAULT 0,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(user_id, character_pack_id, pose_name)
);

CREATE INDEX idx_custom_poses_user ON luna.custom_poses(user_id);
CREATE INDEX idx_custom_poses_pack ON luna.custom_poses(character_pack_id);
CREATE INDEX idx_custom_poses_public ON luna.custom_poses(is_public) WHERE is_public = TRUE;

-- Store outfit variations
CREATE TABLE IF NOT EXISTS luna.character_outfits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    character_pack_id UUID NOT NULL REFERENCES luna.character_packs(id) ON DELETE CASCADE,
    
    outfit_name TEXT NOT NULL,
    outfit_category TEXT,  -- "casual", "formal", "fantasy", etc.
    outfit_prompt TEXT NOT NULL,  -- Full prompt for img2img
    
    -- Generated images for this outfit (per pose)
    outfit_images JSONB,  -- { "pose_id": "image_url", ... }
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(character_pack_id, outfit_name)
);

CREATE INDEX idx_character_outfits_pack ON luna.character_outfits(character_pack_id);
```

---

### Technical Advantages

**Why This Works:**

1. **3D Pose Control = Perfect Consistency**
   - No more "AI can't draw hands" - 3D model has perfect anatomy
   - Depth + OpenPose ControlNets enforce exact pose
   - Canny edges add subtle guidance

2. **Multi-ControlNet = Precise Output**
   - OpenPose: Body structure (80% weight)
   - Depth: Spatial relationships (60% weight)
   - Canny: Edge details (40% weight)
   - Combined = AI has no choice but to match pose

3. **HyperLoRA = Identity Preservation**
   - Character face/body features locked in
   - Works across all poses/outfits
   - Img2Img reinforces consistency

4. **Custom Nodes = Speed + Simplicity**
   - Single node replaces 20+ default nodes
   - Reduced latency (fewer node transitions)
   - Built-in error handling
   - Automatic preprocessing

5. **Img2Img Outfit Changes = Fast Iterations**
   - User tries 10 outfits in 2 minutes
   - No need to regenerate entire character
   - Preserves pose perfectly

---

### Implementation Complexity

**Your Experience Advantages:**
- ✅ ComfyUI workflow expertise
- ✅ Custom node development experience
- ✅ ControlNet integration knowledge
- ✅ 3D → 2D projection understanding
- ✅ Img2Img workflow mastery

**Technical Challenges (Mitigated by Experience):**
- 3D model integration (Three.js + Mixamo rigs are well-documented)
- Multi-ControlNet balancing (you've done this before)
- Custom node optimization (you can streamline to 3-5 nodes total)
- Pose JSON → 2D rendering (standard OpenGL/WebGL projection)

**Timeline Estimate:**
- **Phase 1:** Custom ComfyUI nodes (3 nodes) - 1 week
- **Phase 2:** 3D pose editor UI (Three.js integration) - 2 weeks
- **Phase 3:** Full-body prompt builder - 3 days
- **Phase 4:** Outfit variation system - 1 week
- **Phase 5:** Testing + optimization - 1 week

**Total:** 5-6 weeks for complete full-body system

---

## Conclusion

The **Character Creation Studio** transforms character design from a tedious prompt-engineering task into a **visual, game-like experience** with **guaranteed high-quality, consistent results**.

**Core Innovations (Phase 1):**
✅ Visual design system (not text prompts)  
✅ HyperLoRA + ControlNet for perfect consistency  
✅ Expression transfer from generic templates  
✅ Parallel processing for fast generation (6-8 seconds per expression)  
✅ High-margin monetization (99%+ profit margin)  
✅ One-time purchase model (not subscription)  

**Advanced Innovations (Phase 2 - Full-Body):**
✅ Interactive 3D pose editor with custom posing  
✅ Multi-ControlNet generation (OpenPose + Depth + Canny)  
✅ Img2Img outfit variations (instant outfit changes)  
✅ Custom ComfyUI nodes (3-5 nodes vs 20+)  
✅ Full-body character packs ($24.99-$69.99 tiers)  
✅ Commercial use licensing for premium tiers  

**Competitive Advantages:**
1. **Only platform** with visual character creator + guaranteed expression consistency
2. **Only platform** with 3D-powered pose control for AI generation
3. **Only platform** with instant outfit changing (Img2Img)
4. **One-time purchase** vs subscription (user-friendly)
5. **Immediate story integration** (works in conversations instantly)
6. **Professional quality** (up to 1536x2304 resolution)
7. **Scalable revenue** (low compute cost, high selling price)
8. **Defensible moat** (custom ComfyUI nodes + 3D pipeline non-trivial to replicate)

**Market Positioning:**
- **Character.AI:** Chat-only, no images → **LUNA adds visual character creation**
- **NovelAI:** Basic image gen, no consistency → **LUNA guarantees consistency**
- **Daz3D/Character Creator:** Manual 3D modeling, hours of work → **LUNA: AI-powered, minutes**
- **Midjourney/Stable Diffusion:** Inconsistent results → **LUNA: ControlNet + LoRA guarantee**

This positions LUNA as the **premium character design platform** for AI storytelling, with clear monetization and defensible technology (HyperLoRA + Multi-ControlNet + Custom Nodes pipeline is extremely difficult to replicate).

**Revenue Potential (Year 1):**
- 50,000 users create characters (free prototypes)
- 15,000 purchase Expression Packs ($7.99) = $119,850
- 8,000 purchase Full-Body Standard ($24.99) = $199,920
- 3,000 purchase Full-Body Premium ($39.99) = $119,970
- 1,000 purchase Ultimate Studio ($69.99) = $69,990

**Total:** $509,730 revenue, ~$5,000 compute cost = **$504,730 profit** (99% margin)

---

**Document Status:** Complete (Phase 1 + Phase 2 Expansion)  
**Awaiting:** User approval to begin implementation
