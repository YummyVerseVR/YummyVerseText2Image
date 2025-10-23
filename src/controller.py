import torch

from diffusers.pipelines.stable_diffusion.pipeline_output import (
    StableDiffusionPipelineOutput,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)
from PIL import Image
from pylognet.client import LoggingClient, LogLevel


class StableDiffusionController:
    def __init__(self, config: dict, logger: LoggingClient, debug_mode: bool = False):
        self.__debug = debug_mode
        self.__logger = logger
        self.__config = config.get("stable_diffusion", {})
        self.__pipe = StableDiffusionPipeline.from_pretrained(
            self.__config.get("model", "runwayml/stable-diffusion-v1-5"),
            torch_dtype=torch.float16,
        )
        self.__pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.__pipe.scheduler.config
        )

        if self.__config.get("use_lora", False):
            self.__pipe.load_lora_weights(
                self.__config.get("lora", "michecosta/food_mic")
            )
        self.__pipe.enable_attention_slicing()
        self.__pipe.enable_xformers_memory_efficient_attention()
        self.__pipe.enable_model_cpu_offload()
        self.__pipe.fuse_lora()

    def generate(self, prompt: str) -> Image.Image:
        self.__logger.log(
            f"Received generate request with prompt: {prompt}",
            LogLevel.INFO,
        )

        prompt_template = self.__config.get("prompt_template", "{food}")
        negative_prompt = self.__config.get("negative_prompt", "")

        generated = self.__pipe(
            prompt=prompt_template.format(food=prompt),
            negative_prompt=negative_prompt,
            num_inference_steps=200,
            width=512,
            height=512,
        )
        self.__logger.log(
            "Image generation completed",
            LogLevel.INFO,
        )

        if not isinstance(generated, StableDiffusionPipelineOutput):
            return Image.new("RGB", (512, 512))

        image: Image.Image = generated.images[0]

        if self.__debug:
            image.save("debug_dump.png")

        return image
