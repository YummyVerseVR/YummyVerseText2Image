import asyncio
from io import BytesIO
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
    MODEL_ID = "runwayml/stable-diffusion-v1-5"
    LORA_MODEL_ID = "michecosta/food_mic"
    PROMPT_TEMPLATE = "A high quality food photo of a {}, 1 {}, centered composition, isolated, front view, white background"
    NEGATIVE_PROMPT = "plate, dish, bowl, utensils, table, fork, spoon, chopsticks, text, hands, multiple objects"

    def __init__(
        self,
        db_endpoint: str,
        model_server_endpoint: str,
        debug: bool,
        disable_SD: bool = True,
    ):
        self.__debug = debug
        self.__router = APIRouter()
        self.__app = FastAPI()

        self.__disable_SD = disable_SD
        self.__db_endpoint = db_endpoint
        self.__model_server_endpoint = model_server_endpoint

        self.__pipe = StableDiffusionPipeline.from_pretrained(
            App.MODEL_ID,
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
        self.__router.add_api_route("/ping", self.ping, methods=["GET"])

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
        file = {"file": ("image.png", image, "image/png")}
        data = {"user_id": user_id}

        if self.__debug:
            print(f"Uploading image to {self.__db_endpoint}/save/image")
            with open("debug_image.png", "wb") as f:
                f.write(image.getbuffer())
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
        print(f"[LOG] Received generate request with prompt: {prompt}")

        if not self.__disable_SD:
            generated = self.__pipe(
                prompt=App.PROMPT_TEMPLATE.format(prompt, prompt),
                negative_prompt=App.NEGATIVE_PROMPT,
                num_inference_steps=200,
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
            gen_buf = BytesIO()
            image.save(gen_buf, format="png")
            gen_buf.seek(0)
            asyncio.create_task(self.__call_model_generator(user_id, buf))
            asyncio.create_task(self.__upload_image(user_id, gen_buf))

            return JSONResponse(
                status_code=200,
                content={"message": "Image generated and uploaded successfully"},
            )
        else:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "External image injection is enabled. Use /send to continue."
                },
            )

    async def ping(self) -> JSONResponse:
        return JSONResponse(status_code=200, content={"message": "pong"})
