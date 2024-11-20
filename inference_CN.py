from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from pipeline_controlnet_inpaint import StableDiffusionControlNetInpaintPipeline

from transformers import AutoTokenizer, PretrainedConfig
import torch

controlnet = ControlNetModel.from_pretrained('./controlnet')
def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

        
tokenizer = AutoTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        subfolder="tokenizer",
        use_fast=False,
    )

sd_inpainting_model_name = "stabilityai/stable-diffusion-2-inpainting"
# import correct text encoder class
text_encoder_cls = import_model_class_from_model_name_or_path(sd_inpainting_model_name, None)

# Load scheduler and models
noise_scheduler = DDPMScheduler.from_pretrained(sd_inpainting_model_name, subfolder="scheduler")
text_encoder = text_encoder_cls.from_pretrained(
    sd_inpainting_model_name, subfolder="text_encoder", revision=None
)
vae = AutoencoderKL.from_pretrained(sd_inpainting_model_name, subfolder="vae", revision=None)
unet = UNet2DConditionModel.from_pretrained(
    sd_inpainting_model_name, subfolder="unet", revision=None
)

weight_dtype = torch.float32
pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    sd_inpainting_model_name,
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    controlnet=controlnet,
    safety_checker=None,
    revision=None,
    torch_dtype=weight_dtype,
)
pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline = pipeline.to('cuda')
pipeline.set_progress_bar_config(disable=True)



from PIL import Image, ImageOps
import requests
from io import BytesIO

def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

seed = 0
image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/Granja_comary_Cisne_-_Escalavrado_e_Dedo_De_Deus_ao_fundo_-Teres%C3%B3polis.jpg/2560px-Granja_comary_Cisne_-_Escalavrado_e_Dedo_De_Deus_ao_fundo_-Teres%C3%B3polis.jpg'
response = requests.get(image_url)
img = Image.open(BytesIO(response.content))
img = resize_with_padding(img, (512, 512))

seed = 13
mask = ImageOps.invert(fg_mask)
img = resize_with_padding(img, (512, 512))
generator = torch.Generator(device='cuda').manual_seed(seed)
prompt = 'A dark swan in a bedroom'
cond_scale = 0.0
with torch.autocast("cuda"):
    sd2_image = pipeline(
        prompt=prompt, image=img, mask_image=mask, control_image=mask, num_images_per_prompt=1, generator=generator, num_inference_steps=20, guess_mode=False, controlnet_conditioning_scale=cond_scale
    ).images[0]
sd2_image


mask = ImageOps.invert(fg_mask)
img = resize_with_padding(img, (512, 512))
generator = torch.Generator(device='cuda').manual_seed(seed)
cond_scale = 1.0
with torch.autocast("cuda"):
    controlnet_image = pipeline(
        prompt=prompt, image=img, mask_image=mask, control_image=mask, num_images_per_prompt=1, generator=generator, num_inference_steps=20, guess_mode=False, controlnet_conditioning_scale=cond_scale
    ).images[0]
controlnet_image