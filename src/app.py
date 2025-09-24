import asyncio
import requests

from io import BytesIO
from fastapi import FastAPI, APIRouter, Form
from fastapi.responses import JSONResponse

from controller import StableDiffusionController


class App:
    def __init__(
        self,
        config: dict,
        debug: bool,
    ):
        self.__debug = debug
        self.__config = config
        self.__endpoints = self.__config.get("endpoints", {})
        self.__control_endpoint = self.__endpoints.get(
            "control", "http://localhost:8000"
        )
        self.__model_endpoint = self.__endpoints.get("model", "http://localhost:8003")
        self.__stable_diffusion_controller = StableDiffusionController(config)
        self.__router = APIRouter()
        self.__app = FastAPI()

        self.__setup_routes()

    def __setup_routes(self):
        self.__router.add_api_route("/generate", self.generate_image, methods=["POST"])
        self.__router.add_api_route("/ping", self.ping, methods=["GET"])

    async def __call_model_generator(self, user_id: str, image: BytesIO):
        file = {"file": image}
        data = {"user_id": user_id}

        if self.__debug:
            print(
                f"[DEBUG] Calling model generator at {self.__model_endpoint}/generate is skipped in debug mode."
            )
            return

        requests.post(
            f"{self.__model_endpoint}/generate",
            files=file,
            data=data,
        )

    async def __upload_image(self, user_id: str, image: BytesIO):
        file = {"file": ("image.png", image, "image/png")}
        data = {"user_id": user_id}

        if self.__debug:
            print(
                f"[DEBUG] Uploading image to {self.__control_endpoint}/save/image is skipped in debug mode."
            )
            with open("debug_image.png", "wb") as f:
                f.write(image.getbuffer())
            return

        requests.post(
            f"{self.__control_endpoint}/save/image",
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

        image = await self.__stable_diffusion_controller.generate(prompt)

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

    async def ping(self) -> JSONResponse:
        return JSONResponse(status_code=200, content={"message": "pong"})
