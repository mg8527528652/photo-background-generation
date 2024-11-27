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
import os
import json
import cv2
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


def crop_image_to_nearest_8_multiple(pil_image):
    """
    Crops a PIL image to the nearest multiple of 8 pixels in both dimensions.
    
    Args:
        pil_image: PIL.Image object
        
    Returns:
        PIL.Image: Cropped image with dimensions that are multiples of 8
    """
    # Get image dimensions
    width, height = pil_image.size
    
    # Calculate the deltas (amount to crop)
    delta_height = height % 8
    delta_width = width % 8
    
    # Calculate crop box coordinates (left, top, right, bottom)
    left = delta_width // 2
    top = delta_height // 2
    right = width - (delta_width - delta_width // 2)
    bottom = height - (delta_height - delta_height // 2)
    # Crop and return the image
    return pil_image.crop((left, top, right, bottom))

def crop_image_to_nearest_8_multiple_cv2(image):
    if len(image.shape) == 2:
        height, width = image.shape
    else:   
        height, width, _ = image.shape
    # calculate the delta
    delta_height = height % 8
    delta_width = width % 8
    # crop the image by delta_height / 2, from both top and bottom, and delta_width / 2, from both left and right
    cropped_image = image[delta_height//2:height-(delta_height - delta_height//2), delta_width//2:width-(delta_width - delta_width//2)  ]
    return cropped_image
def setup_pipeline ( controlnet_path, device='cuda'):
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




def resize_with_aspect_ratio(image, width=None, height=None):
    """
    Resize a PIL image maintaining aspect ratio to specified width or height.
    At least one of width or height must be specified.
    
    Args:
        image: PIL.Image object
        width: Target width in pixels, or None
        height: Target height in pixels, or None
        
    Returns:
        PIL.Image: Resized image
    """
    if width is None and height is None:
        raise ValueError("At least one of width or height must be specified")
        
    original_width, original_height = image.size
    
    if width is not None:
        # Calculate height based on width
        aspect_ratio = original_height / original_width
        new_height = int(aspect_ratio * width)
        new_width = width
    else:
        # Calculate width based on height
        aspect_ratio = original_width / original_height
        new_width = int(aspect_ratio * height)
        new_height = height
    
    return image.resize((new_width, new_height), Image.LANCZOS)  # LANCZOS is the modern replacement for ANTIALIAS

def convert4ch_to_3ch(image_4ch):
    # replace background by black and keep the foreground
    image_3ch, mask = image_4ch[:, :, :3], image_4ch[:, :, 3]
    mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask = mask.astype(float) / 255.0
    black_bg = np.zeros((image_3ch.shape[0], image_3ch.shape[1], 3), dtype=np.uint8)
    image_3ch = black_bg * (1 - mask) + image_3ch * mask
    return image_3ch



def overlay_foreground(image_path, generated_image_path):
    # Load images
    img = cv2.imread(image_path, -1)
    generated_image = cv2.imread(generated_image_path, -1)
    
    if img is None or generated_image is None:
        raise ValueError("One of the image paths is invalid or the image could not be loaded.")
    
    if img.shape[2] < 4:
        raise ValueError("The input image must have an alpha channel.")
    
    # Resize the input image to match the dimensions of the generated image
    img = cv2.resize(img, (generated_image.shape[1], generated_image.shape[0]))
    
    # Extract the 3-channel RGB and alpha mask
    img_3ch, mask = img[:, :, :3], img[:, :, 3]
    
    # Normalize the mask
    mask = mask.astype(np.float32) / 255.0
    mask = np.expand_dims(mask, axis=-1)  # Convert to 3D for broadcasting
    
    # Combine the images using the mask
    overlaid_image = generated_image * (1 - mask) + mask * img_3ch
    
    # Ensure the pixel values are within the valid range
    overlaid_image = np.clip(overlaid_image, 0, 255).astype(np.uint8)
    # overlaid_image = cv2.cvtColor(overlaid_image, cv2.COLOR_RGB2BGR)
    # img_3ch = cv2.cvtColor(img_3ch, cv2.COLOR_RGB2BGR)
    # overlaid_image = np.hstack([img_3ch, overlaid_image])
    return overlaid_image


def generate_controlnet_image(image_path, pipeline, prompt, guidance_scale,  img, mask, height, width, device, seed, cond_scale, save_path=None):
        """
        Generate image using ControlNet pipeline
        Args:
            pipeline: ControlNet pipeline
            prompt: Text prompt for generation
            img: Input image
            mask: Mask image
            height: Output image height
            width: Output image width 
            device: Device to run inference on
            seed: Random seed
            cond_scale: ControlNet conditioning scale
            save_path: Optional path to save generated image
        Returns:
            Generated image and overlaid result
        """
        generator = torch.Generator(device=device).manual_seed(seed)
        negative_prompt="octane, random artifacts, extra legs, unwanted elements,bad anatomy, extra fingers, bad fingers, missing fingers, worst hands, improperly holding objects, cropped, blurry, low quality, bad hands, missing legs, missing arms, extra fingers, cg, 3d, unreal, error, out of frame, Cartoon, CGI, Render, 3D, Artwork, Illustration, 3D render, Cinema 4D, Artstation, Octane render, Painting, Oil painting, Anime, 2D, Sketch, <BadDream:1>, by <bad-artist:1>, <UnrealisticDream:1>, <bad_prompt_version2:1>, by <bad-artist-anime:1>, <easynegative:1>",

        with torch.autocast(device):
            controlnet_image = pipeline(
                prompt=prompt,
                image=img,
                mask_image=mask,
                control_image=mask,
                guidance_scale = guidance_scale,
                height = height,
                negative_prompt = negative_prompt[0],
                width= width,
                num_images_per_prompt=1,
                strength=1.0,
                generator=generator,
                num_inference_steps=40,
                guess_mode=False,
                controlnet_conditioning_scale=cond_scale
            ).images[0]
            if save_path:
                controlnet_image.save(save_path)        

        # overlay the foreground of original image with original image, ands keep background as generated image
        # Convert fg_mask to proper transparency mask format
        
        over = overlay_foreground(image_path, save_path)
        
        return controlnet_image, over


def generate_image(
    image_path,
    prompt,
    controlnet_path,
    save_path=None,
    seed=4321,
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

    # img = resize_with_padding(img, (768, 768))
    width, height = img.size
    # resize the largest dim to 768, while maintaqing aspect ratio
    if width > height:
        img = resize_with_aspect_ratio(img, width=768)
    elif height > width:
        img = resize_with_aspect_ratio(img, height=768)
    else:
        img = resize_with_aspect_ratio(img, width=768)
    img = crop_image_to_nearest_8_multiple(img)

    width, height = img.size
    
    print(width, height)
    # Generate foreground mask
    # if image has transparent background, use mode='base' extract mask and convert it RGB
    if img.mode != 'RGBA':
        remover = Remover(mode='base')
        fg_mask = remover.process(img, type='map')
        mask = ImageOps.invert(fg_mask)
    else:
        fg_mask = img.split()[-1].convert("RGB")
        mask = ImageOps.invert(fg_mask)
    
    controlnet_image1, over1 = generate_controlnet_image(
        image_path, pipeline, prompt, 10, img, mask, height, width, device, seed, 0.9, save_path
    )
    # controlnet_image3, over3 = generate_controlnet_image(
    #     image_path,pipeline, prompt, 10, img, mask, height, width, device, seed, 0.75, save_path
    # )
    # controlnet_image4, over4 = generate_controlnet_image(
    #     image_path,pipeline, prompt, 10, img, mask, height, width, device, seed, 0.85, save_path
    # )
    # controlnet_image5, over5 = generate_controlnet_image(
    #     image_path,pipeline, prompt, 10, img, mask, height, width, device, seed, 0.9, save_path
    # )
    # controlnet_image6, over6 = generate_controlnet_image(
    #     image_path,pipeline, prompt, 10, img, mask, height, width, device, seed, 1.0, save_path
    # )
    # over = np.hstack([over1, over3, over4, over5, over6])
    # if save_path:
    #     over.save(save_path)
    
    
    
    # save the overlaid image
    save_path_overlay = save_path.replace('.png', '_overlay.png')
    # cv2.imwrite(save_path_overlay, over1)
    cv2.imwrite(save_path, over1)
        # Generate and return foreground mask
    # controlnet_fg_mask = remover.process(controlnet_image, type='map')
    
    # return controlnet_image6

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
    controlnet_pa = '/home/ubuntu/Desktop/mayank_gaur/controlnet-model'
    images_path = '/home/ubuntu/Desktop/mayank_gaur/BENCHMARK_DATASET/masks_sorted'
    prompts_json_path = '/home/ubuntu/Desktop/mayank_gaur/BENCHMARK_DATASET/bg_prompts'
    save_pat = '/home/ubuntu/Desktop/mayank_gaur/photo-background-generation/outputs-v5-diff-re_91k_cn_0.6_75_0.85_0.9_1-neg_prompt'
    os.makedirs(save_pat, exist_ok=True)
    ckpt_list = ['91000']
    # ckpt_list = ['70000','82000', '75000', '91000']
    for ckpt in ckpt_list:
        try:
            controlnet_path = os.path.join(controlnet_pa, 'checkpoint-' + ckpt, 'controlnet')
            print(controlnet_path)
            save_path = os.path.join(save_pat, 'checkpoint-' + ckpt)
            os.makedirs(save_path, exist_ok=True)   
            for image_name  in os.listdir(images_path):
                try:
                    prompt_name = os.path.join(prompts_json_path, image_name.split('.')[0] + '.json')
                    with open(prompt_name, 'r') as f:
                        prompt = json.load(f)['bg_label']
                    image_path = os.path.join(images_path, image_name)
                    generated_image = generate_image(
                        image_path,
                        prompt,
                        controlnet_path,
                        save_path=os.path.join(save_path, os.path.basename(image_path))
                    )
                    
                except Exception as e:
                    print(e)
                    continue
        except Exception as e:
            print(e)
            continue

