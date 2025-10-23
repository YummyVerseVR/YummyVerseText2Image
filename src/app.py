import requests

from io import BytesIO
from fastapi import FastAPI, APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pylognet.client import LoggingClient, LogLevel
from concurrent.futures import ThreadPoolExecutor

from controller import StableDiffusionController


class UserRequest(BaseModel):
    user_id: str
    prompt: str


class App:
    def __init__(
        self,
        config: dict,
        debug: bool,
        logging: bool,
    ):
        self.__debug = debug
        self.__config = config
        self.__endpoints = self.__config.get("endpoints", {})
        self.__control_endpoint = self.__endpoints.get(
            "control", "http://localhost:8000"
        )
        self.__model_endpoint = self.__endpoints.get("model", "http://localhost:8003")
        self.__logger_endpoint = self.__endpoints.get("logger", "http://localhost:8003")

        self.__logger = LoggingClient(
            "YummyT2IServer",
            self.__logger_endpoint,
            disable=not logging,
        )

        self.__stable_diffusion_controller = StableDiffusionController(
            config,
            self.__logger,
            self.__debug,
        )
        self.__executor = ThreadPoolExecutor()
        self.__router = APIRouter()
        self.__app = FastAPI()

        self.__setup_routes()

    def __setup_routes(self):
        self.__router.add_api_route("/generate", self.generate, methods=["POST"])
        self.__router.add_api_route("/ping", self.ping, methods=["GET"])

    def __call_model_generator(self, user_id: str, image: BytesIO):
        file = {"file": image}
        data = {"user_id": user_id}

        if self.__debug:
            self.__logger.log(
                f"Calling model generator at {self.__model_endpoint}/generate is skipped in debug mode.",
                LogLevel.DEBUG,
            )
            return

        requests.post(
            f"{self.__model_endpoint}/generate",
            files=file,
            data=data,
        )

    def __upload_image(self, user_id: str, image: BytesIO):
        file = {"file": ("image.png", image, "image/png")}
        data = {"user_id": user_id}

        if self.__debug:
            self.__logger.log(
                f"Uploading image to {self.__control_endpoint}/save/image is skipped in debug mode.",
                LogLevel.DEBUG,
            )
            with open("debug_image.png", "wb") as f:
                f.write(image.getbuffer())
            return

        requests.post(
            f"{self.__control_endpoint}/save/image",
            files=file,
            data=data,
        )

    def __generate(self, request: UserRequest) -> None:
        image = self.__stable_diffusion_controller.generate(request.prompt)

        buf = BytesIO()
        image.save(buf, format="png")
        buf.seek(0)
        gen_buf = BytesIO()
        image.save(gen_buf, format="png")
        gen_buf.seek(0)
        self.__call_model_generator(request.user_id, buf)
        self.__upload_image(request.user_id, gen_buf)

    def get_app(self) -> FastAPI:
        self.__app.include_router(self.__router)
        return self.__app

    # /generate
    async def generate(self, request: UserRequest) -> JSONResponse:
        self.__logger.log(
            f"Received generate request with prompt: {request.prompt}",
            LogLevel.INFO,
        )
        self.__executor.submit(self.__generate, request)

        return JSONResponse(
            status_code=200,
            content={"message": "Image generation is submitted."},
        )

    async def ping(self) -> JSONResponse:
        return JSONResponse(status_code=200, content={"message": "pong"})
