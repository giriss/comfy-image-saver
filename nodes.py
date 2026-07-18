import os
from datetime import datetime
from typing import Any, Iterator

import numpy as np
from PIL import Image

import folder_paths


class FormatTemplate:
    """Handles template string formatting with caching."""
    
    TIMESTAMP_CACHE_TTL = 1  # seconds
    
    def __init__(self):
        self._timestamp_cache: dict[str, tuple[str, float]] = {}
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize whitespace in text strings."""
        return " ".join(text.split())
    
    def get_timestamp(self, fmt: str) -> str:
        """Get timestamp with caching to avoid repeated datetime calls."""
        now = datetime.now()
        current_time = now.timestamp()
        
        if fmt in self._timestamp_cache:
            cached_value, cached_time = self._timestamp_cache[fmt]
            if current_time - cached_time < self.TIMESTAMP_CACHE_TTL:
                return cached_value
        
        try:
            formatted = now.strftime(fmt)
        except (ValueError, TypeError):
            formatted = now.strftime("%Y-%m-%d-%H%M%S")
        
        self._timestamp_cache[fmt] = (formatted, current_time)
        return formatted
    
    def format_template(self, template: str, metadata: dict[str, Any], time_format: str) -> str:
        """Format template string with metadata values."""
        if not template:
            return self.get_timestamp(time_format)
        
        replacements = {
            "%date": self.get_timestamp("%Y-%m-%d"),
            "%time": self.get_timestamp(time_format),
            "%model": self.normalize_text(str(metadata.get("model", "unknown"))),
            "%seed": self.normalize_text(str(metadata.get("seed", "unknown"))),
            "%counter": str(metadata.get("counter", 0)),
        }
        
        result = template
        for key, value in replacements.items():
            result = result.replace(key, value)
        
        return result


class MetadataExtractor:
    """Extracts metadata from nested dictionaries."""
    
    MODEL_KEYS = frozenset({"model", "model_name", "ckpt_name"})
    SEED_KEYS = frozenset({"seed"})
    
    @staticmethod
    def flatten_dict(data: Any) -> Iterator[tuple[str, Any]]:
        """Recursively flatten nested dictionaries and lists."""
        if isinstance(data, dict):
            for key, value in data.items():
                yield key, value
                yield from MetadataExtractor.flatten_dict(value)
        elif isinstance(data, (list, tuple)):
            for item in data:
                yield from MetadataExtractor.flatten_dict(item)
    
    @classmethod
    def extract_value(cls, data: Any, keys: frozenset[str]) -> Any | None:
        """Extract first matching value from nested structure."""
        if data is None:
            return None
        for key, value in cls.flatten_dict(data):
            if key in keys:
                return value
        return None
    
    @classmethod
    def extract_metadata(cls, extra_pnginfo: dict | None, prompt: dict | None) -> dict[str, Any]:
        """Extract and compile metadata from available sources."""
        model = cls.extract_value(extra_pnginfo, cls.MODEL_KEYS)
        if model is None:
            model = cls.extract_value(prompt, cls.MODEL_KEYS)
        
        seed = cls.extract_value(extra_pnginfo, cls.SEED_KEYS)
        if seed is None:
            seed = cls.extract_value(prompt, cls.SEED_KEYS)
        
        return {
            "model": str(model) if model is not None else "unknown",
            "seed": str(seed) if seed is not None else "unknown",
            "counter": 0
        }


class ImageWriter:
    """Handles the actual file writing operations."""
    
    SAVE_CONFIGS = {
        "png": {"compress_level": 4, "optimize": True},
        "jpeg": {"quality": 95, "optimize": True},
        "webp": {"quality": 95, "optimize": True, "method": 6},
    }
    
    @staticmethod
    def tensor_to_pil(tensor: Any) -> Image.Image:
        """Convert tensor to PIL Image efficiently."""
        array = (tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(array)
    
    @classmethod
    def save_batch(
        cls,
        images: Any,
        output_path: str,
        filename_base: str,
        extension: str
    ) -> list[str]:
        """Save batch of images to disk."""
        saved_files = []
        batch_size = images.size(0)
        use_index = batch_size > 1
        
        config = cls.SAVE_CONFIGS.get(extension, {"optimize": True})
        
        for idx, tensor in enumerate(images):
            img = cls.tensor_to_pil(tensor)
            
            filename = (
                f"{filename_base}_{idx + 1:02d}.{extension}"
                if use_index
                else f"{filename_base}.{extension}"
            )
            
            file_path = os.path.join(output_path, filename)
            img.save(file_path, **config)
            saved_files.append(filename)
        
        return saved_files


class BatchImageSaver:
    """ComfyUI node for saving batches of images with flexible naming."""
    
    def __init__(self):
        self.output_dir = folder_paths.output_directory
        self._save_counter = 0
        self._template_formatter = FormatTemplate()
        self._time_format = "%Y-%m-%d-%H%M%S"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename": ("STRING", {"default": "%time_%seed", "multiline": False}),
                "path": ("STRING", {"default": "", "multiline": False}),
                "extension": (["png", "jpeg", "webp"],),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "ImageSaverTools"
    
    def save_images(self, images, filename, path, extension, prompt=None, extra_pnginfo=None):
        """Save images with formatted filenames and paths."""
        self._save_counter += 1
        
        # Extract and update metadata
        metadata = MetadataExtractor.extract_metadata(extra_pnginfo, prompt)
        metadata["counter"] = self._save_counter
        
        # Format paths
        filename_base = self._template_formatter.format_template(
            filename, metadata, self._time_format
        )
        relative_path = self._template_formatter.format_template(
            path, metadata, self._time_format
        )
        
        # Prepare output directory
        output_path = os.path.join(self.output_dir, relative_path)
        if output_path.strip() != "":
            os.makedirs(output_path, exist_ok=True)
        else:
            output_path = self.output_dir
        
        # Write images
        saved_files = ImageWriter.save_batch(
            images, output_path, filename_base, extension.lower()
        )
        
        # Prepare UI response
        subfolder = os.path.normpath(relative_path)
        if subfolder == ".":
            subfolder = ""
        
        return {
            "ui": {
                "images": [
                    {"filename": name, "subfolder": subfolder, "type": "output"}
                    for name in saved_files
                ]
            }
        }


NODE_CLASS_MAPPINGS = {
    "Batch Image Save": BatchImageSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Batch Image Save": "Batch Image Save"
}
