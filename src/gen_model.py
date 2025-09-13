from diffusers import DiffusionPipeline
import torch
class GenModel:
    def __init__(self, pipe):
        self.base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        self.base.unet = torch.compile(self.base.unet, mode="reduce-overhead", fullgraph=True)
        self.base.enable_model_cpu_offload()
        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.refiner.unet = torch.compile(self.refiner.unet, mode="reduce-overhead", fullgraph=True)
        self.refiner.enable_model_cpu_offload()

        # Define how many steps and what % of steps to be run on each experts (80/20) here
        n_steps = 40
        high_noise_frac = 0.8

        


    def generate(self, prompt: str):
        prompt = prompt
        negative_prompt = "table, chopsticks, spoon, restaurant, kitchen, background, scenery, person, hands, fingers, text, watermark, logo, signature, blurry, worst quality, low quality, jpeg artifacts"
        # run both experts
        image = self.base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=self.n_steps,
            denoising_end=self.high_noise_frac,
            output_type="latent",
            resolution=1024,
        ).images
        image = self.refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=self.n_steps,
            denoising_start=self.high_noise_frac,
            image=image,
        ).images[0]

        return image