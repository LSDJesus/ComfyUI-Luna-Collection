# Luna Collection Prompt Templates

This directory contains pre-built prompt templates optimized for use with Luna Collection nodes. Each template is designed to work seamlessly with our prompt processing and enhancement features.

## Template Categories

### 🎨 Art Styles
- `artistic.json` - Various artistic styles and techniques
- `photorealistic.json` - Professional photography styles
- `digital-art.json` - Digital art and illustration styles

### 🌟 Themes
- `landscapes.json` - Natural landscapes and environments
- `portraits.json` - Character portraits and expressions
- `architecture.json` - Buildings and architectural styles

### ⚡ Specialized
- `product.json` - Product photography and commercial
- `fantasy.json` - Fantasy and sci-fi themes
- `abstract.json` - Abstract and conceptual art

## Usage

### Basic Usage
```python
# Load template
with open('assets/prompts/artistic.json', 'r') as f:
    template = json.load(f)

# Use with Luna Unified Prompt Processor
processor = LunaUnifiedPromptProcessor()
enhanced_prompt = processor.process(
    text=template['base_prompt'],
    style_template=template['style_enhancements'],
    quality_level='quality'
)
```

### Template Structure
```json
{
  "name": "Professional Photography",
  "description": "High-quality photography style template",
  "version": "1.0.0",
  "base_prompt": "professional photograph, studio lighting, sharp focus",
  "style_enhancements": [
    "masterpiece",
    "best quality",
    "highly detailed",
    "professional lighting"
  ],
  "negative_prompts": [
    "blurry",
    "low quality",
    "distorted",
    "amateur"
  ],
  "recommended_settings": {
    "cfg": 7.5,
    "steps": 30,
    "sampler": "dpmpp_2m",
    "scheduler": "karras"
  },
  "tags": ["photography", "professional", "studio"]
}
```

## Template Files

### artistic.json
```json
{
  "name": "Artistic Styles",
  "description": "Collection of artistic painting and drawing styles",
  "version": "1.0.0",
  "categories": {
    "oil_painting": {
      "base_prompt": "oil painting, classical art style, rich textures, brush strokes visible",
      "style_enhancements": [
        "masterpiece",
        "oil on canvas",
        "impressionist",
        "fine art",
        "museum quality"
      ]
    },
    "watercolor": {
      "base_prompt": "watercolor painting, soft edges, flowing colors, artistic technique",
      "style_enhancements": [
        "watercolor illustration",
        "soft watercolor",
        "artistic style",
        "hand painted",
        "vibrant colors"
      ]
    },
    "digital_illustration": {
      "base_prompt": "digital illustration, clean lines, vibrant colors, modern art",
      "style_enhancements": [
        "digital art",
        "illustration",
        "sharp focus",
        "vibrant",
        "concept art"
      ]
    }
  }
}
```

### photorealistic.json
```json
{
  "name": "Photorealistic Styles",
  "description": "Professional photography and photorealistic rendering styles",
  "version": "1.0.0",
  "categories": {
    "studio_portrait": {
      "base_prompt": "professional studio portrait, perfect lighting, sharp focus, high resolution",
      "style_enhancements": [
        "masterpiece",
        "photorealistic",
        "studio lighting",
        "professional photography",
        "8k resolution"
      ]
    },
    "landscape": {
      "base_prompt": "breathtaking landscape photography, golden hour, professional composition",
      "style_enhancements": [
        "landscape photography",
        "professional",
        "highly detailed",
        "sharp focus",
        "dramatic lighting"
      ]
    }
  }
}
```

## Custom Template Creation

### Template Guidelines
1. **Clarity**: Use clear, descriptive names and descriptions
2. **Completeness**: Include all necessary components (base prompt, enhancements, negatives)
3. **Optimization**: Test templates with Luna's prompt processing
4. **Compatibility**: Ensure compatibility with different model types
5. **Documentation**: Include usage examples and recommended settings

### Template Validation
```python
def validate_template(template: dict) -> bool:
    """Validate prompt template structure."""
    required_fields = ['name', 'description', 'version', 'base_prompt']

    for field in required_fields:
        if field not in template:
            return False

    # Validate prompt quality
    if len(template['base_prompt']) < 10:
        return False

    return True
```

## Integration Examples

### With Luna Unified Prompt Processor
```python
from luna_collection.nodes import LunaUnifiedPromptProcessor

# Load and use template
processor = LunaUnifiedPromptProcessor()

with open('assets/prompts/photorealistic.json', 'r') as f:
    templates = json.load(f)

style = templates['categories']['studio_portrait']
enhanced = processor.process(
    text=f"{style['base_prompt']}, young woman, elegant dress",
    enable_preprocessing=True,
    enable_enhancement=True,
    enable_styling=True,
    quality_level='quality'
)
```

### Batch Processing with Templates
```python
# Process multiple prompts with different templates
templates = ['artistic', 'photorealistic', 'digital_art']
subjects = ['portrait', 'landscape', 'still_life']

for template_name in templates:
    for subject in subjects:
        with open(f'assets/prompts/{template_name}.json', 'r') as f:
            template = json.load(f)

        # Generate variations
        prompt = f"{template['base_prompt']}, {subject}"
        # Process with Luna nodes...
```

## Best Practices

### Template Design
- **Modularity**: Keep templates focused on specific styles or themes
- **Flexibility**: Design templates that work with various subjects
- **Quality**: Test templates extensively before publishing
- **Documentation**: Include clear usage instructions and examples

### Performance Optimization
- **Caching**: Cache processed templates for repeated use
- **Batching**: Process multiple prompts together when possible
- **Memory**: Monitor memory usage with large template collections
- **Validation**: Always validate templates before use

### Maintenance
- **Versioning**: Use semantic versioning for template updates
- **Testing**: Maintain test cases for template compatibility
- **Updates**: Regularly update templates based on user feedback
- **Backup**: Keep backups of working template configurations

## Contributing

### Adding New Templates
1. Follow the established JSON structure
2. Test templates with multiple models and settings
3. Include comprehensive documentation
4. Add appropriate tags and categories
5. Submit via pull request with test results

### Template Review Process
- **Validation**: Automated validation of template structure
- **Testing**: Manual testing with various workflows
- **Performance**: Performance impact assessment
- **Compatibility**: Cross-model compatibility verification

## Support

For template-related support:
- Check existing templates for examples
- Review the [Node Reference](../guides/node-reference.md)
- Join the [Discord Community](https://discord.gg/luna-collection)
- Submit issues on [GitHub](https://github.com/LSDJesus/ComfyUI-Luna-Collection/issues)

---

*Templates are regularly updated based on community feedback and new model capabilities. Check for updates regularly.*