# Import required libraries for diffusion models and image processing
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from pipeline_controlnet_inpaint import StableDiffusionControlNetInpaintPipeline
from transformers import AutoTokenizer, PretrainedConfig
from transparent_background import Remover
import torch
import requests
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np

# Helper function to import the correct text encoder based on model architecture
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

# Utility function to resize images while maintaining aspect ratio
def resize_with_padding(img, expected_size):
    """
    Resize image to expected size while maintaining aspect ratio and adding padding if necessary
    Args:
        img: PIL Image to resize
        expected_size: Tuple of (width, height)
    Returns:
        Padded PIL Image of expected size
    """
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

 # Function to calculate object expansion between reference and predicted masks
def obj_expansion(mask_ref, mask_pred):
    """
    Calculate the difference in object area between reference and predicted masks
    Args:
        mask_ref: Reference mask
        mask_pred: Predicted mask
    Returns:
        Float representing the difference in area coverage
    """
    mask_pred = np.array(mask_pred)
    mask_ref = np.array(mask_ref)
    
    area_pred = np.mean(mask_pred > 127)
    area_ref = np.mean(mask_ref > 127)
    
    expansion = area_pred - area_ref
    return expansion

def setup_pipeline(controlnet_path, device='cuda'):
    """
    Set up the Stable Diffusion ControlNet pipeline
    Args:
        controlnet_path: Path to trained ControlNet model
        device: Device to run inference on
    Returns:
        Configured pipeline
    """
    controlnet = ControlNetModel.from_pretrained(controlnet_path)
    
    sd_inpainting_model_name = "stabilityai/stable-diffusion-2-inpainting"
    text_encoder_cls = import_model_class_from_model_name_or_path(sd_inpainting_model_name, None)
    
    # Load components
    tokenizer = AutoTokenizer.from_pretrained(
        sd_inpainting_model_name, subfolder="tokenizer", use_fast=False,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(sd_inpainting_model_name, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        sd_inpainting_model_name, subfolder="text_encoder", revision=None
    )
    vae = AutoencoderKL.from_pretrained(sd_inpainting_model_name, subfolder="vae", revision=None)
    unet = UNet2DConditionModel.from_pretrained(
        sd_inpainting_model_name, subfolder="unet", revision=None
    )

    # Configure pipeline
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
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    
    return pipeline

def generate_image(
    image_path,
    prompt,
    controlnet_path,
    save_path=None,
    seed=13,
    cond_scale=1.0,
    device='cuda'
):
    """
    Generate image using ControlNet inpainting
    Args:
        image_path: Path or URL to input image
        prompt: Text prompt for generation
        controlnet_path: Path to trained ControlNet model
        save_path: Optional path to save generated image
        seed: Random seed for reproducibility
        cond_scale: ControlNet conditioning scale
        device: Device to run inference on
    Returns:
        Generated image and its foreground mask
    """
    # Set up pipeline
    pipeline = setup_pipeline(controlnet_path, device)
    
    # Load and prepare input image
    if image_path.startswith('http'):
        response = requests.get(image_path)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(image_path)
    
    img = resize_with_padding(img, (512, 512))
    
    # Generate foreground mask
    # if image has transparent background, use mode='base' extract mask and convert it RGB
    if img.mode != 'RGBA':
        remover = Remover(mode='base')
        fg_mask = remover.process(img, type='map')
        mask = ImageOps.invert(fg_mask)
    else:
        mask = img.split()[-1].convert("RGB")
        mask = ImageOps.invert(mask)
    
    # Generate image
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.autocast(device):
        controlnet_image = pipeline(
            prompt=prompt,
            image=img,
            mask_image=mask,
            control_image=mask,
            num_images_per_prompt=1,
            strength=1.0,
            generator=generator,
            num_inference_steps=20,
            guess_mode=False,
            controlnet_conditioning_scale=cond_scale
        ).images[0]
    
    # Save if path provided
    if save_path:
        controlnet_image.save(save_path)
    
    # Generate and return foreground mask
    controlnet_fg_mask = remover.process(controlnet_image, type='map')
    
    return controlnet_image, controlnet_fg_mask

def calculate_expansion(original_mask, generated_mask):
    """
    Calculate expansion between original and generated masks
    Args:
        original_mask: Reference foreground mask
        generated_mask: Generated image foreground mask
    Returns:
        Expansion metric
    """
    return obj_expansion(original_mask, generated_mask)

if __name__ == "__main__":
    # Example usage
    controlnet_path = '/home/ubuntu/Desktop/mayank_gaur/photo-background-generation/controlnet-model/checkpoint-35000/controlnet'
    image_url = 'https://images.fineartamerica.com/images/artworkimages/mediumlarge/1/swan-pond-david-stasiak.jpg'
    prompt = 'A dark swan in a bedroom'
    
    generated_image, generated_mask = generate_image(
        image_url,
        prompt,
        controlnet_path,
        save_path='output.png'
    )