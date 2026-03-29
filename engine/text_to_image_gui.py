import torch
from diffusers import (
    StableDiffusionXLPipeline, StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline
)
import os
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import time
import logging
import random
import gc
import warnings

# Import Compel z obsługą SDXL
try:
    from compel import Compel, ReturnedEmbeddingsType
    COMPEL_AVAILABLE = True
except ImportError:
    COMPEL_AVAILABLE = False
    print("Warning: Compel not installed. Advanced prompt features disabled.")

# Import scentralizowanych modułów narzędziowych i nowych silników
from metadata_utils import MetadataWriter
from model_manager import model_manager
from file_utils import output_manager
from scheduler_utils import SchedulerManager
from v_prediction_utils import VPredictionDetector
from batch_utils import batch_generator
from config_gui import DEFAULT_CONFIG
from engine_t2i import TextToImageEngine
from engine_i2i import ImageToImageEngine

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ignoruj warningi deprecation
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")

class CustomProgressBar(ttk.Progressbar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.style = ttk.Style()
        self.configure_style()
    
    def configure_style(self):
        self.style.configure(
            "Green.Horizontal.TProgressbar",
            background='#2ecc71',
            troughcolor='#34495e',
            bordercolor='#34495e',
            lightcolor='#2ecc71',
            darkcolor='#27ae60'
        )
        self.configure(style="Green.Horizontal.TProgressbar")

class StableDiffusionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stable Diffusion Text-to-Image Generator")
        self.root.geometry("1000x850")
        self.root.minsize(850, 700) # Minimalny rozmiar okna
        self.root.configure(bg='#2b2b2b')
        
        self.txt2img_pipeline = None
        self.img2img_pipeline = None
        self.compel = None
        self.is_sdxl = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.is_generating = False
        self.gallery_data = []
        
        # Inicjalizacja silników
        self.t2i_engine = TextToImageEngine()
        self.i2i_engine = ImageToImageEngine()
        
        self.setup_ui()
        self.update_device_info()
        self.scan_models()
        
    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#2b2b2b')
        style.configure('TLabelframe', background='#2b2b2b', foreground='white')
        style.configure('TLabelframe.Label', background='#2b2b2b', foreground='#87ceeb', font=('Segoe UI', 10, 'bold'))
        style.configure('TLabel', background='#2b2b2b', foreground='white')
        style.configure('TButton', background='#4a4a4a', foreground='white', borderwidth=1)
        style.map('TButton', background=[('active', '#5a5a5a')])
        style.configure('TEntry', fieldbackground='#3c3c3c', foreground='white')
        style.configure('TCombobox', fieldbackground='#3c3c3c', foreground='white')
        style.configure('TCheckbutton', background='#2b2b2b', foreground='white')

        # Główne wagi siatki dla okna
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # Wagi dla głównego kontenera - sprawia, że okno reaguje na skalowanie
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(7, weight=1) # Galeria będzie się rozszerzać w pionie
        
        # --- Wybór modelu ---
        ttk.Label(main_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, pady=5)
        model_frame = ttk.Frame(main_frame)
        model_frame.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        model_frame.columnconfigure(0, weight=1)
        
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, state="readonly")
        self.model_combo.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_selected)
        
        refresh_btn = ttk.Button(model_frame, text="⟳", width=3, command=self.scan_models)
        refresh_btn.grid(row=0, column=1, padx=5)
        
        # --- Prompt ---
        ttk.Label(main_frame, text="Prompt:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.prompt_text = scrolledtext.ScrolledText(main_frame, height=3, width=50, bg='#3c3c3c', fg='white', insertbackground='white')
        self.prompt_text.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        if COMPEL_AVAILABLE:
            compel_info_frame = ttk.Frame(main_frame)
            compel_info_frame.grid(row=2, column=1, columnspan=2, sticky='w', padx=5)
            ttk.Label(compel_info_frame, text="✓ Compel active.", foreground="lightgreen").pack(side=tk.LEFT, padx=(0, 10))
            ttk.Label(compel_info_frame, text="Use (word)++ to strengthen, [word] to weaken.", foreground="#888").pack(side=tk.LEFT)
        
        # --- Negative Prompt ---
        ttk.Label(main_frame, text="Negative Prompt:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.negative_prompt_text = scrolledtext.ScrolledText(main_frame, height=2, width=50, bg='#3c3c3c', fg='white', insertbackground='white')
        self.negative_prompt_text.insert("1.0", "worst quality, low quality, displeasing, text, watermark, bad anatomy")
        self.negative_prompt_text.grid(row=3, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # --- Parametry Generacji (Zgrupowane i skalowalne) ---
        params_frame = ttk.LabelFrame(main_frame, text=" Generation Parameters ", padding="10 5")
        params_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Rozłożenie parametrów aby lepiej oddychały
        for i in range(8):
            params_frame.columnconfigure(i, weight=1 if i % 2 != 0 else 0)

        ttk.Label(params_frame, text="Width:").grid(row=0, column=0, padx=5, sticky=tk.E)
        self.width_var = tk.IntVar(value=DEFAULT_CONFIG.default_width)
        ttk.Spinbox(params_frame, from_=512, to=DEFAULT_CONFIG.max_image_size[0], increment=64, textvariable=self.width_var, width=8).grid(row=0, column=1, padx=5, sticky=tk.W)
        
        ttk.Label(params_frame, text="Height:").grid(row=0, column=2, padx=5, sticky=tk.E)
        self.height_var = tk.IntVar(value=DEFAULT_CONFIG.default_height)
        ttk.Spinbox(params_frame, from_=512, to=DEFAULT_CONFIG.max_image_size[1], increment=64, textvariable=self.height_var, width=8).grid(row=0, column=3, padx=5, sticky=tk.W)
        
        ttk.Label(params_frame, text="Steps:").grid(row=0, column=4, padx=5, sticky=tk.E)
        self.steps_var = tk.IntVar(value=DEFAULT_CONFIG.default_steps)
        ttk.Spinbox(params_frame, from_=1, to=100, textvariable=self.steps_var, width=8).grid(row=0, column=5, padx=5, sticky=tk.W)
        
        ttk.Label(params_frame, text="Guidance:").grid(row=0, column=6, padx=5, sticky=tk.E)
        self.guidance_var = tk.DoubleVar(value=DEFAULT_CONFIG.default_guidance)
        ttk.Spinbox(params_frame, from_=1.0, to=20.0, increment=0.1, textvariable=self.guidance_var, width=8).grid(row=0, column=7, padx=5, sticky=tk.W)
        
        ttk.Label(params_frame, text="Seed:").grid(row=1, column=0, padx=5, pady=10, sticky=tk.E)
        self.seed_var = tk.StringVar()
        ttk.Entry(params_frame, textvariable=self.seed_var, width=15).grid(row=1, column=1, padx=5, pady=10, sticky=tk.W)
        
        ttk.Label(params_frame, text="Sampler:").grid(row=1, column=2, padx=5, pady=10, sticky=tk.E)
        self.scheduler_var = tk.StringVar(value=DEFAULT_CONFIG.default_scheduler)
        # Zwiększona szerokość z 10 na 25, aby pomieścić długie nazwy z DPM++
        scheduler_combo = ttk.Combobox(params_frame, textvariable=self.scheduler_var, values=SchedulerManager.get_available_schedulers(), width=25, state="readonly")
        scheduler_combo.grid(row=1, column=3, padx=5, pady=10, sticky=tk.W)
        
        ttk.Label(params_frame, text="Batch:").grid(row=1, column=4, padx=5, pady=10, sticky=tk.E)
        self.batch_var = tk.IntVar(value=1)
        ttk.Spinbox(params_frame, from_=1, to=20, textvariable=self.batch_var, width=8).grid(row=1, column=5, padx=5, pady=10, sticky=tk.W)
        
        self.v_prediction_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(params_frame, text="V-Prediction", variable=self.v_prediction_var).grid(row=1, column=6, columnspan=2, padx=5, pady=10, sticky=tk.W)
        
        # --- Przyciski ---
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=5, column=0, columnspan=3, pady=10)
        self.generate_btn = ttk.Button(buttons_frame, text="Generate", command=self.start_generation_thread)
        self.generate_btn.grid(row=0, column=0, padx=5)
        self.stop_btn = ttk.Button(buttons_frame, text="Stop", command=self.stop_generation, state='disabled')
        self.stop_btn.grid(row=0, column=1, padx=5)
        
        # --- Pasek Postępu ---
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        progress_frame.columnconfigure(0, weight=1)
        self.progress = CustomProgressBar(progress_frame, mode='determinate', maximum=100)
        self.progress.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.progress_var, foreground="lightblue").grid(row=1, column=0, pady=(5, 0))
        
        # --- Galeria Obrazów ---
        gallery_frame = ttk.Frame(main_frame)
        gallery_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.N, tk.S, tk.E, tk.W), pady=10)
        gallery_frame.rowconfigure(0, weight=1)
        gallery_frame.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(gallery_frame, bg="#2b2b2b", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        scrollbar = ttk.Scrollbar(gallery_frame, orient="horizontal", command=self.canvas.xview)
        scrollbar.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(xscrollcommand=scrollbar.set)
        self.image_container = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.image_container, anchor="nw")
        self.image_container.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        
        # --- Pasek Stanu ---
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        status_frame.columnconfigure(0, weight=1)
        self.device_label = ttk.Label(status_frame, text="", foreground="lightblue")
        self.device_label.grid(row=0, column=0, sticky=tk.W)
        self.model_info_var = tk.StringVar(value="Model: Not selected")
        ttk.Label(status_frame, textvariable=self.model_info_var, foreground="lightyellow").grid(row=1, column=0, sticky=tk.W)
        self.output_info_var = tk.StringVar(value=f"Output: {os.path.abspath(output_manager.base_output_dir)}")
        ttk.Label(status_frame, textvariable=self.output_info_var, foreground="lightgreen").grid(row=2, column=0, sticky=tk.W)

    def add_image_to_gallery(self, image: Image.Image, metadata: dict):
        gallery_index = len(self.gallery_data)
        self.gallery_data.append({'image': image, 'metadata': metadata})
        img_copy = image.copy()
        img_copy.thumbnail(DEFAULT_CONFIG.preview_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img_copy)
        image_label = ttk.Label(self.image_container, image=photo, padding=5, cursor="hand2")
        image_label.image = photo
        image_label.pack(side=tk.LEFT, anchor=tk.NW)
        image_label.bind("<Button-3>", lambda event, idx=gallery_index: self.show_context_menu(event, idx))

    def show_context_menu(self, event, gallery_index: int):
        context_menu = tk.Menu(self.root, tearoff=0, bg="#3c3c3c", fg="white")
        context_menu.add_command(label="Upscale (1.5x)", command=lambda: self.start_postprocess_thread('upscale', gallery_index))
        context_menu.add_command(label="Variation (Strength 0.55)", command=lambda: self.start_postprocess_thread('variation', gallery_index))
        try:
            context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            context_menu.grab_release()

    def start_postprocess_thread(self, mode: str, gallery_index: int):
        if self.is_generating:
            messagebox.showwarning("Warning", "Poczekaj na zakończenie bieżącego zadania.")
            return
        self.set_ui_state(False)
        thread = threading.Thread(target=self.run_postprocess_flow, args=(mode, gallery_index))
        thread.daemon = True
        thread.start()

    def run_postprocess_flow(self, mode: str, gallery_index: int):
        self.is_generating = True
        try:
            original_data = self.gallery_data[gallery_index]
            self.progress_var.set(f"Przygotowywanie do zadania: {mode}...")
            self.load_model_if_needed()
            self.load_img2img_pipeline_if_needed()
            self.progress['value'] = 0
            
            # Delegowanie zadania do silnika I2I
            new_image, new_metadata = self.i2i_engine.process(
                mode=mode,
                pipeline=self.img2img_pipeline,
                compel=self.compel,
                is_sdxl=self.is_sdxl,
                original_image=original_data['image'],
                metadata=original_data['metadata'],
                device=self.device,
                dtype=self.torch_dtype
            )

            output_path = output_manager.get_next_output_path(extension="png")
            image_with_metadata, png_info = MetadataWriter.add_metadata_to_image(new_image, new_metadata)
            image_with_metadata.save(output_path, format="PNG", pnginfo=png_info)
            self.add_image_to_gallery(new_image, new_metadata)
            self.output_info_var.set(f"Zapisano: {os.path.basename(output_path)}")
            self.progress['value'] = 100
            self.progress_var.set(f"Zadanie '{mode}' zakończone sukcesem!")
        except Exception as e:
            logger.error(f"Błąd podczas post-processingu: {e}", exc_info=True)
            messagebox.showerror("Error", f"Wystąpił błąd: {e}")
            self.progress_var.set(f"Error: {e}")
        finally:
            self.set_ui_state(True)
            self.cleanup_memory()

    def load_img2img_pipeline_if_needed(self):
        if self.img2img_pipeline:
            return
        if not self.txt2img_pipeline:
            self.progress_var.set("Błąd: najpierw załaduj model text-to-image.")
            return
        self.progress_var.set("Inicjalizacja potoku Image-to-Image...")
        self.root.update_idletasks()
        PipelineClass = StableDiffusionXLImg2ImgPipeline if self.is_sdxl else StableDiffusionImg2ImgPipeline
        self.img2img_pipeline = PipelineClass(vae=self.txt2img_pipeline.vae, text_encoder=self.txt2img_pipeline.text_encoder, tokenizer=self.txt2img_pipeline.tokenizer, unet=self.txt2img_pipeline.unet, scheduler=self.txt2img_pipeline.scheduler, text_encoder_2=getattr(self.txt2img_pipeline, 'text_encoder_2', None), tokenizer_2=getattr(self.txt2img_pipeline, 'tokenizer_2', None)).to(self.device)
        if self.device == "cuda":
            self.img2img_pipeline.enable_vae_tiling()
        self.progress_var.set("Potok Image-to-Image gotowy.")

    def run_generation_flow(self):
        self.is_generating = True
        self.clear_image_gallery()
        try:
            prompt = self.prompt_text.get("1.0", tk.END).strip()
            negative_prompt = self.negative_prompt_text.get("1.0", tk.END).strip()
            model_name = self.model_var.get()
            batch_size = self.batch_var.get()
            seed_str = self.seed_var.get().strip()
            base_seed = int(seed_str) if seed_str.isdigit() else random.randint(0, 2**32 - 1)
            self.load_model_if_needed()
            start_time = time.time()
            for i in range(batch_size):
                if not self.is_generating: break
                progress_info = batch_generator.get_batch_progress(i, batch_size)
                self.progress['value'] = progress_info['percentage']
                self.progress_var.set(f"Generowanie obrazu {i+1}/{batch_size}...")
                self.root.update_idletasks()
                current_seed = base_seed + i
                
                # Delegowanie zadania do silnika T2I
                image = self.t2i_engine.generate(
                    pipeline=self.txt2img_pipeline,
                    compel=self.compel,
                    is_sdxl=self.is_sdxl,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=self.width_var.get(),
                    height=self.height_var.get(),
                    steps=self.steps_var.get(),
                    guidance=self.guidance_var.get(),
                    seed=current_seed,
                    device=self.device,
                    dtype=self.torch_dtype
                )
                
                metadata = MetadataWriter.create_metadata(prompt=prompt, negative_prompt=negative_prompt, width=self.width_var.get(), height=self.height_var.get(), steps=self.steps_var.get(), guidance_scale=self.guidance_var.get(), seed=current_seed, scheduler=self.scheduler_var.get(), model_name=model_name, v_prediction=self.v_prediction_var.get(), batch_index=i if batch_size > 1 else None)
                
                # Odbieramy teraz krotkę: obraz oraz obiekt PngInfo
                image_with_metadata, png_info = MetadataWriter.add_metadata_to_image(image, metadata)
                
                # Wymuszamy format PNG w ścieżce
                output_path = output_manager.get_next_output_path(extension="png")
                
                # Zapisujemy jako PNG z podaniem pnginfo
                image_with_metadata.save(output_path, format="PNG", pnginfo=png_info)
                self.add_image_to_gallery(image, metadata)
                self.output_info_var.set(f"Zapisano: {os.path.basename(output_path)}")
            total_time = time.time() - start_time
            self.progress['value'] = 100
            self.progress_var.set(f"Batch zakończony! Wygenerowano {batch_size} obrazów w {total_time:.2f}s.")
        except Exception as e:
            logger.error(f"Błąd generacji: {e}", exc_info=True)
            messagebox.showerror("Error", f"Wystąpił błąd: {e}")
            self.progress_var.set(f"Error: {e}")
        finally:
            self.set_ui_state(True)
            self.cleanup_memory()

    def load_model_if_needed(self):
        model_name = self.model_var.get()
        model_path = model_manager.get_model_path(model_name)
        v_prediction = self.v_prediction_var.get()
        current_model_path = getattr(self.txt2img_pipeline, '_model_path', None)
        current_v_prediction = getattr(self.txt2img_pipeline, '_v_prediction', None)
        if self.txt2img_pipeline and model_path == current_model_path and v_prediction == current_v_prediction:
            self.progress_var.set("Model jest już załadowany.")
            return
        self.progress_var.set("Ładowanie modelu...")
        self.root.update_idletasks()
        self.cleanup_memory()
        self.txt2img_pipeline = None
        self.img2img_pipeline = None
        model_info = model_manager.get_model_info(model_name)
        self.is_sdxl = model_info['is_sdxl']
        scheduler = SchedulerManager.create_scheduler(self.scheduler_var.get(), v_prediction=v_prediction, is_sdxl=self.is_sdxl)
        PipelineClass = StableDiffusionXLPipeline if self.is_sdxl else StableDiffusionPipeline
        self.txt2img_pipeline = PipelineClass.from_single_file(model_path, torch_dtype=self.torch_dtype, use_safetensors=model_info['extension'] == '.safetensors', scheduler=scheduler).to(self.device)
        if self.device == "cuda":
            self.txt2img_pipeline.enable_vae_tiling()
        if COMPEL_AVAILABLE:
            if self.is_sdxl:
                self.compel = Compel(tokenizer=[self.txt2img_pipeline.tokenizer, self.txt2img_pipeline.tokenizer_2], text_encoder=[self.txt2img_pipeline.text_encoder, self.txt2img_pipeline.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True], device=self.device, truncate_long_prompts=False)
            else:
                self.compel = Compel(tokenizer=self.txt2img_pipeline.tokenizer, text_encoder=self.txt2img_pipeline.text_encoder, returned_embeddings_type=ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED, device=self.device, truncate_long_prompts=False)
        try:
            self.txt2img_pipeline.enable_xformers_memory_efficient_attention()
        except Exception:
            print("xformers not available.")
        self.txt2img_pipeline._model_path = model_path
        self.txt2img_pipeline._v_prediction = v_prediction
        self.progress_var.set(f"Model załadowany{' z V-Prediction' if v_prediction else ''}")
        self.cleanup_memory()

    def clear_image_gallery(self):
        self.gallery_data.clear()
        for widget in self.image_container.winfo_children():
            widget.destroy()

    def scan_models(self):
        try:
            models = model_manager.scan_models()
            if models:
                self.model_combo['values'] = models
                if not self.model_var.get() or self.model_var.get() not in models:
                    self.model_var.set(models[0])
                self.on_model_selected()
                self.progress_var.set(f"Znaleziono {len(models)} modeli")
            else:
                self.model_combo['values'] = ["Nie znaleziono modeli"]
                self.model_var.set("Nie znaleziono modeli")
                self.progress_var.set("Brak modeli w folderze /models")
        except Exception as e:
            self.progress_var.set(f"Błąd skanowania modeli: {e}")

    def on_model_selected(self, event=None):
        model_name = self.model_var.get()
        if not model_manager.model_exists(model_name): return
        is_v_pred = VPredictionDetector.is_v_prediction_model(model_name)
        self.v_prediction_var.set(is_v_pred)
        info_text = f"Model: {model_name}" + (" (V-Prediction)" if is_v_pred else "")
        self.model_info_var.set(info_text)
        self.progress_var.set(f"Wybrano: {model_name}")

    def update_device_info(self):
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            info = f"Device: {gpu_name} ({memory:.1f}GB) - {self.torch_dtype}"
        else:
            info = f"Device: CPU - {self.torch_dtype}"
        self.device_label.config(text=info)

    def start_generation_thread(self):
        if self.is_generating:
            messagebox.showwarning("Warning", "Poczekaj na zakończenie bieżącego zadania.")
            return
        if not model_manager.model_exists(self.model_var.get()):
            messagebox.showerror("Error", "Proszę wybrać prawidłowy model.")
            return
        if not self.prompt_text.get("1.0", tk.END).strip():
            messagebox.showerror("Error", "Proszę wprowadzić prompt.")
            return
        self.set_ui_state(False)
        thread = threading.Thread(target=self.run_generation_flow)
        thread.daemon = True
        thread.start()

    def stop_generation(self):
        self.is_generating = False
        self.progress_var.set("Zatrzymywanie generacji...")

    def set_ui_state(self, enabled: bool):
        state = 'normal' if enabled else 'disabled'
        self.generate_btn.config(state=state)
        self.stop_btn.config(state='disabled' if enabled else 'normal')
        self.is_generating = not enabled

    def cleanup_memory(self):
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def on_closing(self):
        self.is_generating = False
        self.cleanup_memory()
        del self.txt2img_pipeline
        del self.img2img_pipeline
        del self.compel
        self.cleanup_memory()
        self.root.destroy()

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    root = tk.Tk()
    app = StableDiffusionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
