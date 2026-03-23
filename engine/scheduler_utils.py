from diffusers import (
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    DEISMultistepScheduler,
    UniPCMultistepScheduler
)

from typing import Dict, Any


class SchedulerManager:
    """Manager do zarządzania schedulerami z obsługą V-Prediction (format A1111-WebUI)"""

    SCHEDULER_CONFIGS = {
        # --- DPM++ 2M ---
        "DPM++ 2M": {
            "class": DPMSolverMultistepScheduler,
            "repo": "runwayml/stable-diffusion-v1-5"
        },
        "DPM++ 2M Karras": {
            "class": DPMSolverMultistepScheduler,
            "repo": "runwayml/stable-diffusion-v1-5",
            "kwargs": {"use_karras_sigmas": True}
        },
        "DPM++ 2M SDE": {
            "class": DPMSolverMultistepScheduler,
            "repo": "runwayml/stable-diffusion-v1-5",
            "kwargs": {"algorithm_type": "sde-dpmsolver++"}
        },
        "DPM++ 2M SDE Karras": {
            "class": DPMSolverMultistepScheduler,
            "repo": "runwayml/stable-diffusion-v1-5",
            "kwargs": {"use_karras_sigmas": True, "algorithm_type": "sde-dpmsolver++"}
        },

        # --- DPM++ 2S a ---
        "DPM++ 2S a": {
            "class": DPMSolverSinglestepScheduler,
            "repo": "runwayml/stable-diffusion-v1-5"
        },
        "DPM++ 2S a Karras": {
            "class": DPMSolverSinglestepScheduler,
            "repo": "runwayml/stable-diffusion-v1-5",
            "kwargs": {"use_karras_sigmas": True}
        },

        # --- DPM++ SDE ---
        "DPM++ SDE": {
            "class": DPMSolverSinglestepScheduler,
            "repo": "runwayml/stable-diffusion-v1-5"
        },
        "DPM++ SDE Karras": {
            "class": DPMSolverSinglestepScheduler,
            "repo": "runwayml/stable-diffusion-v1-5",
            "kwargs": {"use_karras_sigmas": True}
        },

        # --- DPM2 ---
        "DPM2": {
            "class": KDPM2DiscreteScheduler,
            "repo": "runwayml/stable-diffusion-v1-5"
        },
        "DPM2 Karras": {
            "class": KDPM2DiscreteScheduler,
            "repo": "runwayml/stable-diffusion-v1-5",
            "kwargs": {"use_karras_sigmas": True}
        },
        "DPM2 a": {
            "class": KDPM2AncestralDiscreteScheduler,
            "repo": "runwayml/stable-diffusion-v1-5"
        },
        "DPM2 a Karras": {
            "class": KDPM2AncestralDiscreteScheduler,
            "repo": "runwayml/stable-diffusion-v1-5",
            "kwargs": {"use_karras_sigmas": True}
        },

        # --- Euler ---
        "Euler": {
            "class": EulerDiscreteScheduler,
            "repo": "runwayml/stable-diffusion-v1-5"
        },
        "Euler a": {
            "class": EulerAncestralDiscreteScheduler,
            "repo": "runwayml/stable-diffusion-v1-5"
        },

        # --- Heun ---
        "Heun": {
            "class": HeunDiscreteScheduler,
            "repo": "runwayml/stable-diffusion-v1-5"
        },

        # --- LMS ---
        "LMS": {
            "class": LMSDiscreteScheduler,
            "repo": "runwayml/stable-diffusion-v1-5"
        },
        "LMS Karras": {
            "class": LMSDiscreteScheduler,
            "repo": "runwayml/stable-diffusion-v1-5",
            "kwargs": {"use_karras_sigmas": True}
        },

        # --- Inne (Zgodnie ze zrzutami ekranu - Diffusers) ---
        "DEIS": {
            "class": DEISMultistepScheduler,
            "repo": "runwayml/stable-diffusion-v1-5"
        },
        "UniPC": {
            "class": UniPCMultistepScheduler,
            "repo": "runwayml/stable-diffusion-v1-5"
        }
    }

    @classmethod
    def create_scheduler(cls,
                         scheduler_type: str,
                         v_prediction: bool = False,
                         is_sdxl: bool = False) -> Any:
        
        # Mapowanie niewrażliwe na wielkość liter
        scheduler_map = {k.lower(): k for k in cls.SCHEDULER_CONFIGS.keys()}
        normalized_type = scheduler_type.lower()

        # Domyślny fallback jeśli nie znaleziono
        if normalized_type not in scheduler_map:
            actual_scheduler_key = "Euler a"
        else:
            actual_scheduler_key = scheduler_map[normalized_type]

        config = cls.SCHEDULER_CONFIGS[actual_scheduler_key]
        SchedulerClass = config["class"]

        prediction_type = "v_prediction" if v_prediction else "epsilon"

        repo_id = (
            "stabilityai/stable-diffusion-xl-base-1.0"
            if is_sdxl else config["repo"]
        )

        kwargs = config.get("kwargs", {})

        return SchedulerClass.from_pretrained(
            repo_id,
            subfolder="scheduler",
            prediction_type=prediction_type,
            **kwargs
        )

    @classmethod
    def get_available_schedulers(cls) -> list:
        # Zwracamy listę kluczy (przyjaznych nazw), gotowych do wstawienia do elementu UI
        return list(cls.SCHEDULER_CONFIGS.keys())
