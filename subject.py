import torch
from diffusers.pipelines import FluxPipeline
from src.flux.condition import Condition
from PIL import Image

from src.flux.generate import generate, seed_everything

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")
pipe.load_lora_weights(
    "Yuanshi/OminiControl",
    weight_name=f"omini/subject_512.safetensors",
    adapter_name="subject",
)

# image = Image.open("assets/room_l.png").convert("RGB").resize((512, 512))
# image = Image.open("assets/penguin.jpg").convert("RGB").resize((512, 512))
image = Image.open("assets/mantee.png").convert("RGB").resize((512, 512))

condition = Condition("subject", image, position_delta=(0, -32))

condition = Condition("subject", image, position_delta=(0, -32))

# prompt = "A modern living room with a large gray sectional sofa, a round pendant light, a dining table with two chairs, a gray door, and a white wall."
# prompt = "On Christmas evening, on a crowded sidewalk, this item sits on the road, covered in snow and wearing a Christmas hat."
prompt = "In a serene mangrove estuary, a manatee glides gracefully through the clear, shallow waters. The photo is taken from an underwater side perspective, allowing a full view of the manatee's gray, wrinkled skin and paddle-like tail. The soft, diffuse sunlight filters through the water, creating dappled patterns on the sandy bottom. In the background, seagrasses sway with the gentle current, and small fish dart around the manatee's path. The tranquil scene is enhanced by the shimmering reflections of the surrounding mangrove canopy on the water's surface"

seed_everything(2024)

result_img = generate(
    pipe,
    prompt=prompt,
    conditions=[condition],
    num_inference_steps=8,
    height=512,
    width=512,
    # guidance_scale=0., # https://huggingface.co/docs/diffusers/en/api/pipelines/flux#timestep-distilled
    # max_sequence_length=256, # https://huggingface.co/docs/diffusers/en/api/pipelines/flux#timestep-distilled
).images[0]

concat_image = Image.new("RGB", (1024, 512))
concat_image.paste(image, (0, 0))
concat_image.paste(result_img, (512, 0))
concat_image.save("output_concat_room.png")