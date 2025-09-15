import os
import uuid
import torch
from fastapi import FastAPI, Form
from fastapi.responses import FileResponse
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# モデルとLoRAの準備
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_MODEL = "digiplay/food_mic"  # Hugging Face Hub 上の Food LoRA

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionXLPipeline.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
).to(device)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# LoRA を読み込み & 適用
pipe.load_lora_weights(LORA_MODEL)
pipe.fuse_lora()

# 出力保存用ディレクトリ
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI()


@app.post("/generate")
async def generate_image(
    food_name: str = Form(..., description="食品名（形容詞や副詞を含んでも可）"),
):
    """食品の画像をトップビューで生成し一時保存、ダウンロードIDを返す"""
    prompt = (
        f"{food_name}, realistic food photography, no plate, isolated, "
        f"professional lighting, top view"
    )

    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]

    # 一時ファイルに保存
    file_id = str(uuid.uuid4())
    file_path = os.path.join(OUTPUT_DIR, f"{file_id}.png")
    image.save(file_path)

    return {"file_id": file_id, "download_url": f"/download/{file_id}"}


@app.get("/download/{file_id}")
async def download_image(file_id: str):
    """生成済み画像をダウンロードし、即時削除"""
    file_path = os.path.join(OUTPUT_DIR, f"{file_id}.png")
    if not os.path.exists(file_path):
        return {"error": "File not found or already downloaded"}

    response = FileResponse(
        file_path, media_type="image/png", filename=f"{file_id}.png"
    )

    # ダウンロード後削除
    @response.background
    def cleanup():
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass

    return response
