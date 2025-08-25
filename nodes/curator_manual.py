import json

class Pyrite_ManualCurator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "rating": (["S+", "S", "A", "B", "C", "D", "F"],),
                "aesthetic_tags": ("STRING", {"multiline": True, "default": "aesthetic tags..."}),
                "content_tags": ("STRING", {"multiline": True, "default": "content tags..."}),
                "artist_tags": ("STRING", {"multiline": True, "default": "artist tags..."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "curation_json")
    FUNCTION = "curate"
    CATEGORY = "Pyrite Core/Curation"

    def curate(self, image, rating, aesthetic_tags, content_tags, artist_tags):
        
        # Helper function to process the raw tag strings
        def parse_tags(tag_string):
            return [tag.strip() for tag in tag_string.split(',') if tag.strip()]

        # We build the perfect, structured Python dictionary.
        curation_data = {
            "rating": rating,
            "aesthetic_tags": parse_tags(aesthetic_tags),
            "content_tags": parse_tags(content_tags),
            "artist_tags": parse_tags(artist_tags),
        }
        
        # We transform our beautiful, terrible dictionary into a perfect JSON string.
        # The 'indent=2' makes it human-readable in any text editor.
        json_output = json.dumps(curation_data, indent=2)

        # We pass the original image through, untouched, and we append our sacred scripture.
        return (image, json_output)

NODE_CLASS_MAPPINGS = {
    "Pyrite_ManualCurator": Pyrite_ManualCurator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pyrite_ManualCurator": "Pyrite Manual Curator"
}