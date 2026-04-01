import torch
from compel import Compel

class TextToImageEngine:
    """Silnik odpowiedzialny za generowanie obrazów z tekstu (Text-to-Image)."""

    def generate(self, pipeline, compel: Compel, is_sdxl: bool, prompt: str, negative_prompt: str, width: int, height: int, steps: int, guidance: float, seed: int, device: str, dtype: torch.dtype, clip_skip: int = 1, cfg_rescale: float = 0.0):
        
        generator = torch.Generator(device=device).manual_seed(seed)
        
        if compel:
            return self._generate_with_compel(pipeline, compel, is_sdxl, prompt, negative_prompt, width, height, steps, guidance, generator, device, dtype, cfg_rescale)
        else:
            return self._generate_standard(pipeline, prompt, negative_prompt, width, height, steps, guidance, generator, clip_skip, cfg_rescale)

    def _pad_embeddings(self, prompt_embeds, negative_prompt_embeds, device: str, dtype: torch.dtype):
        """Wyrównuje długość embeddingów."""
        max_length = max(prompt_embeds.shape[1], negative_prompt_embeds.shape[1])

        def pad_tensor(tensor, length):
            if tensor.shape[1] < length:
                padding_size = length - tensor.shape[1]
                padding = torch.zeros((1, padding_size, tensor.shape[2]), device=device, dtype=dtype)
                return torch.cat([tensor, padding], dim=1)
            return tensor

        return pad_tensor(prompt_embeds, max_length), pad_tensor(negative_prompt_embeds, max_length)

    def _generate_with_compel(self, pipeline, compel: Compel, is_sdxl: bool, prompt: str, negative_prompt: str, width: int, height: int, steps: int, guidance: float, generator: torch.Generator, device: str, dtype: torch.dtype, cfg_rescale: float = 0.0):
        
        extra_kwargs = {}
        if cfg_rescale > 0.0:
            extra_kwargs['guidance_rescale'] = cfg_rescale

        with torch.no_grad():
            if is_sdxl:
                prompt_embeds, pooled_prompt_embeds = compel([prompt])
                negative_prompt_embeds, negative_pooled_prompt_embeds = compel([negative_prompt])
                prompt_embeds, negative_prompt_embeds = self._pad_embeddings(prompt_embeds, negative_prompt_embeds, device, dtype)
                
                image = pipeline(
                    prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    width=width, height=height, num_inference_steps=steps, guidance_scale=guidance, generator=generator,
                    **extra_kwargs
                ).images[0]
            else:
                prompt_embeds = compel(prompt)
                negative_prompt_embeds = compel(negative_prompt)
                prompt_embeds, negative_prompt_embeds = self._pad_embeddings(prompt_embeds, negative_prompt_embeds, device, dtype)

                image = pipeline(
                    prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                    width=width, height=height, num_inference_steps=steps, guidance_scale=guidance, generator=generator,
                    **extra_kwargs
                ).images[0]
        return image

    def _generate_standard(self, pipeline, prompt: str, negative_prompt: str, width: int, height: int, steps: int, guidance: float, generator: torch.Generator, clip_skip: int = 1, cfg_rescale: float = 0.0):
        
        extra_kwargs = {}
        if clip_skip > 1:
            extra_kwargs['clip_skip'] = clip_skip
        if cfg_rescale > 0.0:
            extra_kwargs['guidance_rescale'] = cfg_rescale
            
        with torch.no_grad():
            image = pipeline(
                prompt=prompt, negative_prompt=negative_prompt,
                width=width, height=height, num_inference_steps=steps, guidance_scale=guidance, generator=generator,
                **extra_kwargs
            ).images[0]
        return image
