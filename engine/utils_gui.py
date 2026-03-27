import os
import torch
from PIL import Image
import hashlib
from typing import Optional
from pathlib import Path

def validate_model_file(file_path: str) -> bool:
    """Sprawdza poprawność pliku modelu"""
    if not os.path.exists(file_path):
        return False
    
    if not file_path.lower().endswith(('.ckpt', '.safetensors')):
        return False
    
    # Sprawdź rozmiar pliku (minimalny rozmiar dla modelu)
    file_size = os.path.getsize(file_path) / 1024**3  # w GB
    if file_size < 0.1:  # Mniej niż 100MB - prawdopodobnie nie model
        return False
    
    return True

def get_device_info() -> dict:
    """Zwraca informacje o urządzeniu"""
    info = {
        'has_cuda': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'dtype_support': {}
    }
    
    if info['has_cuda']:
        info['current_device'] = torch.cuda.current_device()
        info['device_name'] = torch.cuda.get_device_name(0)
        info['total_memory'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        info['free_memory'] = (torch.cuda.get_device_properties(0).total_memory - 
                              torch.cuda.memory_allocated()) / 1024**3
        
        # Sprawdź wsparcie dla różnych typów danych
        info['dtype_support']['float16'] = True
        info['dtype_support']['bfloat16'] = hasattr(torch, 'bfloat16')
    
    return info

def generate_output_filename(prompt: str, seed: Optional[int] = None) -> str:
    """Generuje nazwę pliku na podstawie prompta i seeda"""
    # Utwórz hash z prompta
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
    
    if seed is not None:
        return f"sd_generation_{prompt_hash}_{seed}.png"
    else:
        return f"sd_generation_{prompt_hash}.png"

def ensure_png_extension(filename: str) -> str:
    """Upewnia się, że nazwa pliku ma rozszerzenie .png"""
    if not filename.lower().endswith('.png') and not filename.lower().endswith('.png'):
        return filename + '.png'
    return filename

def cleanup_memory():
    """Czyści pamięć GPU"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def estimate_generation_time(steps: int, resolution: tuple, is_sdxl: bool = False) -> float:
    """Szacuje czas generowania obrazu"""
    base_time = 0.5  # sekundy na krok
    resolution_factor = (resolution[0] * resolution[1]) / (1024 * 1024)
    sdxl_factor = 1.5 if is_sdxl else 1.0
    
    estimated_time = steps * base_time * resolution_factor * sdxl_factor
    return max(estimated_time, 5.0)  # Minimum 5 sekund
