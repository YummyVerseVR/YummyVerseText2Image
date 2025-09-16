from io import BytesIO
from PIL.Image import Image
from diffusers.pipelines.stable_diffusion.pipeline_output import (
    StableDiffusionPipelineOutput,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)
from fastapi import FastAPI, APIRouter, Form
from fastapi.responses import JSONResponse
import torch
import requests


class App:
    STABLE_DIFFUSION_MODEL_ID = "runwayml/stable-diffusion-v1-5"
    LORA_MODEL_ID = "michecosta/food_mic"
    PROMPT_TEMPLATE = "A high quality photo of a {}, 1 {}, centered composition, isolated, front view, professional food photography, white background"
    NEGATIVE_PROMPT = "plate, dish, bowl, utensils, fork, spoon, chopsticks, table, napkin, text, watermark, hands, multiple objects, low quality, blurry, deformed, disfigured, distorted, ugly"

    def __init__(
        self, db_endpoint: str, model_server_endpoint: str, debug: bool = False
    ):
        self.__debug = debug
        self.__router = APIRouter()
        self.__app = FastAPI()

        self.__db_endpoint = db_endpoint
        self.__model_server_endpoint = model_server_endpoint

        self.__pipe = StableDiffusionPipeline.from_pretrained(
            App.STABLE_DIFFUSION_MODEL_ID,
            torch_dtype=torch.float16,
        )
        self.__pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.__pipe.scheduler.config
        )
        self.__pipe.to("cuda")
        self.__pipe.load_lora_weights(App.LORA_MODEL_ID)
        self.__pipe.enable_attention_slicing()
        self.__pipe.enable_xformers_memory_efficient_attention()
        self.__pipe.enable_model_cpu_offload()
        self.__pipe.fuse_lora()

        self.__setup_routes()

    def __setup_routes(self):
        self.__router.add_api_route("/generate", self.generate_image, methods=["POST"])

    async def __call_model_generator(self, user_id: str, image: BytesIO):
        file = {"file": image}
        data = {"user_id": user_id}

        if self.__debug:
            print(f"Calling model generator at {self.__model_server_endpoint}/generate")
            return

        requests.post(
            f"{self.__model_server_endpoint}/generate",
            files=file,
            data=data,
        )

    async def __upload_image(self, user_id: str, image: BytesIO):
        file = {"file": image}
        data = {"user_id": user_id}

        if self.__debug:
            print(f"Uploading image to {self.__db_endpoint}/save/image")
            return

        requests.post(
            f"{self.__db_endpoint}/save/image",
            files=file,
            data=data,
        )

    def get_app(self) -> FastAPI:
        self.__app.include_router(self.__router)
        return self.__app

    # /generate
    async def generate_image(
        self, user_id: str = Form(...), prompt: str = Form(...)
    ) -> JSONResponse:
        generated = self.__pipe(
            prompt=App.PROMPT_TEMPLATE.format(prompt, prompt),
            negative_prompt=App.NEGATIVE_PROMPT,
            width=512,
            height=512,
        )

        if not isinstance(generated, StableDiffusionPipelineOutput):
            return JSONResponse(
                status_code=500,
                content={"message": "Failed to generate image"},
            )

        image = generated.images[0]
        buf = BytesIO()
        image.save(buf, format="png")
        buf.seek(0)
        await self.__call_model_generator(user_id, buf)
        await self.__upload_image(user_id, buf)

        return JSONResponse(
            status_code=200,
            content={"message": "Image generated and uploaded successfully"},
        )
