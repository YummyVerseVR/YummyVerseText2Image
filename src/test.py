# import torch
# from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

# # ベースモデル（schnell）
# pipe = FluxPipeline.from_pretrained(
#     "black-forest-labs/FLUX.1-schnell",
#     torch_dtype=torch.float16,
#     device_map="balanced",  # 自動で一部をCPUオフロード
#     load_in_8bit=True,  # 8bit量子化（VRAM削減）
# )

# # CPUオフロード有効化（さらにVRAM節約）
# pipe.enable_model_cpu_offload()

# # LoRA を適用（食品特化）
# pipe.load_lora_weights("path/to/food-flux-lora")

# # プロンプト例
# prompt = "Ultra realistic photo of delicious ramen, cinematic lighting, 50mm lens"
# negative_prompt = "blurry, distorted, text, watermark"

# # 推論
# image = pipe(
#     prompt,
#     negative_prompt=negative_prompt,
#     num_inference_steps=20,  # ステップ数を抑える
#     guidance_scale=3.5,  # LoRA併用時は低めが安定
#     width=512,
#     height=512,  # VRAM制約のため解像度を下げる
# ).images[0]

# # 保存
# image.save("food_flux.png")
