from PIL.Image import Image
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)

import torch

# load the base model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

# load the lora weights
pipe.load_lora_weights("michecosta/food_mic")

# apply the lora weights to the model
pipe.fuse_lora()

# generate image
prompt = "A photorealistic image of a curry rice with a side of salad on a white plate"
generated_images = pipe(prompt)
image = None
if isinstance(generated_images, tuple) and isinstance(generated_images[0], list):
    image = generated_images[0][0]

# save image
if image is not None and isinstance(image, Image):
    image.save("generated_food_image.png")
