from fastapi import FastAPI, APIRouter, HTTPException
import base64
from pydantic import BaseModel
from gen_model import GenModel
# HunyuanへのAPIエンドポイント
MODEL_URL= "http://127.0.0.1:8003"

class Item(BaseModel):
    uuid: str
    prompt: str

class App:
    def __init__(self):
        self.app = FastAPI()
        self.router = APIRouter()
        self.setup_routes()

    def setup_routes(self):
        self.router.add_api_route("/", self.read_root, methods=["GET"])
    def get_app(self):
        self.app.include_router(self.router)
        return self.app
    async def read_root(self):
        return {"detail": "A text-to-image generation API using Hunyuan model."}

    async def generate_image(self, item: Item):
        print(item)
        model = GenModel()
        image = model.generate(item.prompt)
        return {"image": image}