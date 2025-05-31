'''
HF_HOME="/home/tiger/.cache/huggingface_re20250424" python examples/inpainting.py
'''

import os
print(os.environ["HF_HOME"])

import torch
from diffusers.pipelines import FluxPipeline
from src.flux.condition import Condition
from PIL import Image

from src.flux.generate import generate, seed_everything

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")

pipe.load_lora_weights(
    "Yuanshi/OminiControl",
    weight_name=f"experimental/fill.safetensors",
    adapter_name="fill",
)

for name, param in pipe.transformer.named_parameters():
    print(f"[Transformer] {name}: {param.shape}, {param.dtype}, {param.requires_grad}")
    
'''
demo svdl
'''

image = Image.open("/mnt/bn/rui-picodata-lf/svdl_data/renderings/rendering_kujiale_official_hdr_1m5/converts_official_512/loader_cubemap_warped_from_persp_gt_depth/20240712/811920/train/cam_sampled_04/cubemap_sdr_warped_mesh/0005_b_512.png").convert("RGB").resize((512, 512))

masked_image = image.copy()
# masked_image.paste((0, 0, 0), (128, 100, 384, 220))

condition = Condition("fill", masked_image)

seed_everything()
result_img = generate(
    pipe,
    prompt="An image of a room",
    conditions=[condition],
).images[0]

concat_image = Image.new("RGB", (1536, 512))
concat_image.paste(image, (0, 0))
concat_image.paste(condition.condition, (512, 0))
concat_image.paste(result_img, (1024, 0))
concat_image.save("examples/svdl_inpainted.png")

'''
demo svdl - debug
'''

image = Image.open("/mnt/bn/rui-picodata-lf/svdl_data/renderings/rendering_kujiale_official_hdr_1m5/converts_official_512/loader_cubemap_warped_from_persp_gt_depth/20240712/811920/train/cam_sampled_04/cubemap_sdr_warped_mesh/0005_b_512.png").convert("RGB").resize((512, 512))

masked_image = image.copy()
masked_image.paste((0, 0, 0), (0, 0, 250, 200))

condition = Condition("fill", masked_image)

seed_everything()
result_img = generate(
    pipe,
    prompt="An image of a room",
    conditions=[condition],
).images[0]

concat_image = Image.new("RGB", (1536, 512))
concat_image.paste(image, (0, 0))
concat_image.paste(condition.condition, (512, 0))
concat_image.paste(result_img, (1024, 0))
concat_image.save("examples/svdl_inpainted_debug.png")

'''
demo svdl 2 
'''

image = Image.open("/mnt/bn/rui-picodata-lf/svdl_data/renderings/rendering_kujiale_official_hdr_1m5/converts_official_512/loader_cubemap/20240712/811920/train/cam_sampled_04/cubemap_sdr_autoexp//0005_b_512.png").convert("RGB").resize((512, 512))

masked_image = image.copy()
masked_image.paste((0, 0, 0), (128, 100, 384, 220))

condition = Condition("fill", masked_image)

seed_everything()
result_img = generate(
    pipe,
    prompt="An image of a room",
    conditions=[condition],
).images[0]

concat_image = Image.new("RGB", (1536, 512))
concat_image.paste(image, (0, 0))
concat_image.paste(condition.condition, (512, 0))
concat_image.paste(result_img, (1024, 0))
concat_image.save("examples/svdl_inpainted_2.png")

'''
demo 1
'''

image = Image.open("assets/monalisa.jpg").convert("RGB").resize((512, 512))

masked_image = image.copy()
masked_image.paste((0, 0, 0), (128, 100, 384, 220))

condition = Condition("fill", masked_image)

seed_everything()
result_img = generate(
    pipe,
    prompt="The Mona Lisa is wearing a white VR headset with 'Omini' written on it.",
    conditions=[condition],
).images[0]

concat_image = Image.new("RGB", (1536, 512))
concat_image.paste(image, (0, 0))
concat_image.paste(condition.condition, (512, 0))
concat_image.paste(result_img, (1024, 0))
concat_image.save("examples/monalisa_inpainted.png")

'''
demo 2
'''
image = Image.open("assets/book.jpg").convert("RGB").resize((512, 512))

w, h, min_dim = image.size + (min(image.size),)
image = image.crop(
    ((w - min_dim) // 2, (h - min_dim) // 2, (w + min_dim) // 2, (h + min_dim) // 2)
).resize((512, 512))


masked_image = image.copy()
masked_image.paste((0, 0, 0), (150, 150, 350, 250))
masked_image.paste((0, 0, 0), (200, 380, 320, 420))

condition = Condition("fill", masked_image)

seed_everything()
result_img = generate(
    pipe,
    prompt="A yellow book with the word 'OMINI' in large font on the cover. The text 'for FLUX' appears at the bottom.",
    conditions=[condition],
).images[0]

concat_image = Image.new("RGB", (1536, 512))
concat_image.paste(image, (0, 0))
concat_image.paste(condition.condition, (512, 0))
concat_image.paste(result_img, (1024, 0))
concat_image.save("examples/book_inpainted.png")