from diffusers import DiffusionPipeline
import torch



# load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)
base.enable_model_cpu_offload()
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)
refiner.enable_model_cpu_offload()

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8

prompt = "spicy chicken"
negative_prompt = "table, chopsticks, spoon, restaurant, kitchen, background, scenery, person, hands, fingers, text, watermark, logo, signature, blurry, worst quality, low quality, jpeg artifacts"
# run both experts
image = base(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
    resolution=1024,
).images
image = refiner(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=image,
).images[0]

image.save("output.png")