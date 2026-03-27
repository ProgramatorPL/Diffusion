import os
import random
import string
from datetime import datetime
from pathlib import Path
from typing import Optional

class OutputManager:
    """Manager do zarządzania plikami wyjściowymi"""
    
    def __init__(self, base_output_dir: str = "output"):
        self.base_output_dir = base_output_dir
        self.generation_counter = self.get_initial_counter()
        self.ensure_base_directory()
    
    def ensure_base_directory(self):
        """Tworzy bazowy katalog wyjściowy jeśli nie istnieje"""
        os.makedirs(self.base_output_dir, exist_ok=True)
    
    def get_initial_counter(self) -> int:
        """Zwraca początkowy licznik na podstawie istniejących plików"""
        today_dir = self.get_today_directory()
        if os.path.exists(today_dir):
            png_files = [f for f in os.listdir(today_dir) if f.endswith('.png')]
            if png_files:
                # Znajdź najwyższy numer
                numbers = []
                for file in png_files:
                    try:
                        num = int(file.split('-')[0])
                        numbers.append(num)
                    except (ValueError, IndexError):
                        continue
                return max(numbers) if numbers else 0
        return 0
    
    def get_today_directory(self) -> str:
        """Zwraca ścieżkę do katalogu z dzisiejszą datą"""
        today = datetime.now().strftime("%Y-%m-%d")
        output_dir = os.path.join(self.base_output_dir, today)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def generate_filename(self, extension: str = "png") -> str:
        """Generuje unikalną nazwę pliku"""
        self.generation_counter += 1
        random_hash = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        return f"{self.generation_counter:04d}-{random_hash}.{extension}"
    
    def get_next_output_path(self, extension: str = "png") -> str:
        """Zwraca pełną ścieżkę do następnego pliku wyjściowego"""
        output_dir = self.get_today_directory()
        filename = self.generate_filename(extension)
        return os.path.join(output_dir, filename)

# Globalna instancja managera
output_manager = OutputManager()

def setup_output_directory():
    """Inicjalizuje katalog wyjściowy i zwraca ścieżkę"""
    return output_manager.get_today_directory()

def get_unique_filename():
    """Zwraca unikalną nazwę pliku"""
    return output_manager.generate_filename()

def get_output_path():
    """Zwraca pełną ścieżkę do pliku wyjściowego"""
    return output_manager.get_next_output_path()
