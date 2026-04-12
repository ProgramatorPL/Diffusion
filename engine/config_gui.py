from dataclasses import dataclass
from typing import Optional

@dataclass
class AppConfig:
    """Konfiguracja aplikacji GUI"""
    default_width: int = 1024
    default_height: int = 1024
    default_steps: int = 20
    default_guidance: float = 7
    default_scheduler: str = "Euler a"
    default_output: str = "output.png"
    max_image_size: tuple = (2048, 2048)
    preview_size: tuple = (400, 400)
    
    @property
    def device_info(self) -> str:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"{gpu_name} ({memory:.1f}GB)"
        return "CPU"

# Domyślna konfiguracja
DEFAULT_CONFIG = AppConfig()
