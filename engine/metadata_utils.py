from PIL import Image
from PIL.PngImagePlugin import PngInfo
import json
from datetime import datetime
from typing import Dict, Any, Optional
import torch

class MetadataWriter:
    """Klasa do zapisywania metadanych rozpoznawanych przez A1111, Forge i CivitAI."""
    
    EXIF_TAGS = {
        'user_comment': 37510,       
        'image_description': 270,   
    }
    
    @classmethod
    def create_metadata(cls, 
                       prompt: str,
                       negative_prompt: str,
                       width: int,
                       height: int,
                       steps: int,
                       guidance_scale: float,
                       seed: Optional[int],
                       scheduler: str,
                       model_name: str,
                       v_prediction: bool,
                       batch_index: Optional[int] = None) -> Dict[str, Any]:
        
        metadata = {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'width': width,
            'height': height,
            'steps': steps,
            'cfg_scale': guidance_scale,
            'sampler': scheduler,
            'model': model_name,
            'v_prediction': v_prediction,
            'generation_date': datetime.now().isoformat(),
            'software': 'StableDiffusionGUI',
            'device': 'CUDA' if torch.cuda.is_available() else 'CPU'
        }
        
        if seed is not None:
            metadata['seed'] = seed
        
        if batch_index is not None:
            metadata['batch_index'] = batch_index
        
        return metadata

    @classmethod
    def metadata_to_civitai_string(cls, metadata: Dict[str, Any]) -> str:
        """Konwertuje metadane na string kompatybilny z A1111-WEBUI / WebUI-Forge."""
        prompt_str = metadata.get('prompt', '').strip()
        negative_prompt_str = f"Negative prompt: {metadata.get('negative_prompt', '').strip()}"
        
        settings = [
            f"Steps: {metadata.get('steps', 'N/A')}",
            f"Sampler: {metadata.get('sampler', 'N/A')}",
            f"CFG scale: {metadata.get('cfg_scale', 'N/A')}",
            f"Seed: {metadata.get('seed', 'N/A')}",
            f"Size: {metadata.get('width', 'N/A')}x{metadata.get('height', 'N/A')}",
            f"Model: {metadata.get('model', 'N/A').split('.')[0]}" 
        ]
        
        # Kluczowe dla CivitAI: jeśli wykryjemy operację i2i, dodajemy Denoising strength
        if 'variation_strength' in metadata:
            settings.append(f"Denoising strength: {metadata.get('variation_strength')}")
        elif 'upscale_factor' in metadata:
            # W pliku engine_i2i.py masz ustawione strength na 0.4 dla trybu upscale
            settings.append("Denoising strength: 0.4")
            
        settings_str = ", ".join(settings)
        
        # Jeśli nie ma negative promptu, pomijamy jego linijkę, żeby struktura była czysta
        if metadata.get('negative_prompt', '').strip():
            return f"{prompt_str}\n{negative_prompt_str}\n{settings_str}"
        else:
            return f"{prompt_str}\n{settings_str}"

    @classmethod
    def add_metadata_to_image(cls, image: Image.Image, metadata: Dict[str, Any]) -> tuple[Image.Image, PngInfo]:
        """
        Zwraca krotkę (obraz_z_exif, obiekt_png_info). 
        Obiekt PngInfo zawiera dane w formacie 'parameters' (wymagane przez CivitAI dla plików PNG).
        """
        image_with_exif = image.copy()
        civitai_string = cls.metadata_to_civitai_string(metadata)
        json_string = json.dumps(metadata, ensure_ascii=False)

        # 1. Format PNG (Standard dla WebUI i CivitAI)
        png_info = PngInfo()
        png_info.add_text("parameters", civitai_string) # Magiczny klucz 'parameters'
        png_info.add_text("Workflow", json_string)      # Opcjonalnie przechowuj surowy JSON 

        # 2. Format EXIF (Awaryjnie / Dla formatu JPEG)
        try:
            exif_data = image_with_exif.getexif()

            # A1111 często zapisuje UserComment z prefiksem UNICODE
            comment_bytes = b"UNICODE\x00" + civitai_string.encode('utf-8')
            exif_data[cls.EXIF_TAGS['user_comment']] = comment_bytes
            exif_data[cls.EXIF_TAGS['image_description']] = json_string

            image_with_exif.info["exif"] = exif_data.tobytes()
        except Exception as e:
            print(f"Warning: Could not write EXIF data. Error: {e}")
            
        return image_with_exif, png_info
