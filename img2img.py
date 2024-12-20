from diffusers import AutoencoderKL, AutoPipelineForImage2Image, ControlNetModel, EulerAncestralDiscreteScheduler, DPMSolverSinglestepScheduler
import torch
from diffusers.utils import load_image
import cv2
# from crop_image_to_nearest_8_multiple import crop_image_to_nearest_8_multiple
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from PIL import ImageOps
import torchvision.transforms as transforms
import json

def load_models(controlnet_path, base_model_path):
    # Initialize ControlNet pipeline
    controlnet = ControlNetModel.from_pretrained(
        controlnet_path, 
        torch_dtype=torch.float16,
        # conditioning_channels = 5
    )
    pipe = AutoPipelineForImage2Image.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin", low_cpu_mem_usage=True)


    # pipe.to('cuda')
    return pipe



def stack_horizontally(imgs):
    # imgs is an array containing PIL images. stack them horizontally to create strips.
    # return a single PIL image.
    img_width = imgs[0].width
    img_height = imgs[0].height
    strip_width = img_width * len(imgs)
    strip_height = img_height
    strip = Image.new('RGB', (strip_width, strip_height))
    for i, img in enumerate(imgs):
        strip.paste(img, (i * img_width, 0))
    return strip

def convert4ch_to_3ch(image_4ch):
    # replace background by black and keep the foreground for PIL RGBA image
    # Split into RGB and alpha channels
    rgb_image = image_4ch.convert('RGB')
    alpha_channel = image_4ch.split()[-1]
    
    # Create black background
    black_bg = Image.new('RGB', image_4ch.size, (0, 0, 0))
    
    # Convert alpha to mask
    mask = Image.new('L', image_4ch.size, 0)
    mask.paste(alpha_channel)
    
    # Composite RGB image over black background using alpha mask
    result = Image.composite(rgb_image, black_bg, mask)
    
    return result

def crop_image_to_nearest_8_multiple(image):
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


def resize_with_padding_pil(img, expected_size, fill = (255, 0, 0)):
        # Get the original dimensions
        w, h = img.size
        desired_w, desired_h = expected_size

        # Calculate scaling factor to maintain aspect ratio
        aspect = w / h
        if aspect > desired_w / desired_h:  # wider than tall
            new_w = desired_w
            new_h = int(new_w / aspect)
        else:  # taller than wide
            new_h = desired_h
            new_w = int(new_h * aspect)

        # Resize image while maintaining aspect ratio
        resized = img.resize((new_w, new_h), Image.LANCZOS)

        # Calculate padding
        delta_w = desired_w - new_w
        delta_h = desired_h - new_h
        padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)

        # Add padding with red color, acc to the channel size
        # Add padding with appropriate fill value based on image mode
        if fill != (255, 0, 0):
            fill = fill
        elif img.mode == 'RGBA':
            fill = (255, 0, 0, 255)  # Red with full opacity
        elif img.mode == 'RGB':
            fill = (255, 0, 0)  # Red with full opacity
        elif img.mode == 'L':
            fill = 255  # White for single channel grayscale
        else:
            raise ValueError(f"Unsupported image mode: {img.mode}")

        padded = ImageOps.expand(resized, padding, fill=fill)

        return padded
    
    
def process_single_image(image, resolution=1024):
    """
    Process a single PIL image similar to the preprocess_train function.
    
    Args:
        image (PIL.Image): Input PIL image
        resolution (int): Target resolution for resizing
    
    Returns:
        tuple: (processed_image_tensor, processed_conditioning_tensor)
    """
    # Define transforms
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    conditioning_image_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Convert to RGB and process main image
    # rgb_image = convert4ch_to_3ch(image)
    rgb_image = image.convert('RGB')
    rgb_image = resize_with_padding_pil(rgb_image, (resolution, resolution))
    image_tensor = image_transforms(rgb_image)
    
    #mask image
    # last channel of image
    mask = image.split()[-1].convert('RGB')
    mask = resize_with_padding_pil(mask, (resolution, resolution), fill=(255, 255, 255))
    # invert mask
    mask = ImageOps.invert(mask)
    mask_tensor = conditioning_image_transforms(mask)
    # binarize mask
    mask_tensor[mask_tensor >= 0.5] = 1
    mask_tensor[mask_tensor < 0.5] = 0
    # add batch dimension
    mask_tensor = mask_tensor.unsqueeze(0)
    # convert mask back to rgb pil image
    mask = mask_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Reshape to (H, W, C)
    mask = (mask * 255).astype(np.uint8)
    mask = Image.fromarray(mask)
    
    # Create and process black image
    black_image = Image.new("L", image.size, (0))
    black_image = resize_with_padding_pil(black_image, (resolution, resolution))
    black_tensor = conditioning_image_transforms(black_image)
    
    # Process conditioning image (RGBA)
    conditioning_image = image.convert("RGBA")
    conditioning_image = resize_with_padding_pil(conditioning_image, (resolution, resolution))
    conditioning_tensor = conditioning_image_transforms(conditioning_image)
    
    # Make alpha channel binary
    alpha_channel = conditioning_tensor[3]
    alpha_channel[alpha_channel >= 0.5] = 1
    alpha_channel[alpha_channel < 0.5] = 0
    conditioning_tensor[3] = alpha_channel
    conditioning_tensor = torch.cat((conditioning_tensor, black_tensor), dim=0)
    
    # Add batch dimension to both tensors
    image_tensor = image_tensor.unsqueeze(0)  # Add this line
    conditioning_tensor = conditioning_tensor.unsqueeze(0)  # Add this line
    # Concatenate black image as 5th channel
    
    return rgb_image, conditioning_tensor, mask

 
def generate_controlled_image(
    image,
    prompt,
    negative_prompt="low quality, bad quality",
    seed=123456,
    num_inference_steps=30,
    guidance_scale=12,
    controlnet_conditioning_scale=1.0,
    strength=1.0,
    pipe=None
):
    # read image and mask
    # image = Image.open(image_path)
    
    image_tensor, conditioning_tensor, mask_tensor = process_single_image(image)
    # Infer
    images = pipe(
            prompt=prompt,
            # mask_image = mask_tensor,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            strength=strength,
            image=image_tensor,
            control_image=conditioning_tensor,
            generator=torch.manual_seed(seed),
            guidance_scale=guidance_scale,
            # aesthetic_score= 10,
            controlnet_conditioning_scale=1.0,    
        ).images
    # Remove padding by checking red channel in conditioning tensor
    conditioning_tensor = conditioning_tensor.squeeze(0)
    mask = conditioning_tensor[-1] != 1.0  # Get mask of non-red pixels
    image_ = images[0]
    # Crop image to remove padding using the mask
    image_array = np.array(image_)
    rows = np.any(mask.cpu().numpy(), axis=1)
    cols = np.any(mask.cpu().numpy(), axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    cropped_image = Image.fromarray(image_array[y_min:y_max+1, x_min:x_max+1])
    images[0] = cropped_image
    # creeate image composite. overlay image foreground on cropped image
    image_composite = Image.composite(image.convert('RGB'), cropped_image, image.split()[-1].convert('L')).convert('RGB')
    return image_composite   #, stack_horizontally([images[0], images2[0], images3[0], images4[0], images5[0] ])


def resize_with_padding_cv2(img, expected_size):
    """Resize a cv2 image maintaining aspect ratio using padding
    
    Args:
        img: cv2/numpy image array with shape (H, W, C)
        expected_size: tuple of (width, height)
    """
    # Get original dimensions
    h, w = img.shape[:2]
    desired_w, desired_h = expected_size

    # Calculate scaling factor to maintain aspect ratio
    aspect = w / h
    if aspect > desired_w / desired_h:  # wider than tall
        new_w = desired_w
        new_h = int(new_w / aspect)
    else:  # taller than wide
        new_h = desired_h
        new_w = int(new_h * aspect)

    # Resize image while maintaining aspect ratio
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Calculate padding
    delta_w = desired_w - new_w
    delta_h = desired_h - new_h
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    # Add padding with red color based on number of channels
    if len(img.shape) == 3:
        if img.shape[2] == 4:  # BGRA
            color = [0, 0, 255, 255]  # Red with full opacity in BGR-A
        else:  # BGR
            color = [0, 0, 255]  # Red in BGR
    else:  # Grayscale
        color = 255  # White for single channel

    padded = cv2.copyMakeBorder(
        resized,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return padded


def overlay_foreground(image_path, generated_image_path):
    # Load images
    img = cv2.imread(image_path, -1)
    generated_image = cv2.imread(generated_image_path, -1)
    
    img = resize_with_padding_cv2(img, (generated_image.shape[1], generated_image.shape[0]))
    
    if img is None or generated_image is None:
        raise ValueError("One of the image paths is invalid or the image could not be loaded.")
    
    if img.shape[2] < 4:
        raise ValueError("The input image must have an alpha channel.")
    
    # # Resize the input image to match the dimensions of the generated image
    # img = cv2.resize(img, (generated_image.shape[1], generated_image.shape[0]))
    
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


if __name__ == "__main__":
    # Example usage
    images_path = "/home/ubuntu/mayank/photo-background-generation/benchmark_dataset/BENCHMARK_DATASET/masks_sorted"
    jpg_path = ""
    prompts_path  = "/home/ubuntu/mayank/photo-background-generation/benchmark_dataset/BENCHMARK_DATASET/bg_prompts"
    save_path = "/home/ubuntu/mayank/cn_img2img_juggernaut_multi_signals"
    base_controlnet_path = "/root/photo-background-generation/ckpts/cn_train_inpaint_sdxl_v2"
    base_model_path = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
    controlnet_models = os.listdir(base_controlnet_path)

    for controlnet_model in controlnet_models:
        try:    
            # if controlnet_model not in ['checkpoint-'+i+'000' for i in ['65', '61', '58', '57', '50', '47', '24', '23', '22']]:
            if controlnet_model not in ['checkpoint-'+i+'000' for i in ['83']]:
                print('skipping')
                continue
            controlnet_path = os.path.join(base_controlnet_path , controlnet_model , "controlnet")
            pipe = load_models(controlnet_path, base_model_path)
            print(controlnet_path)
            for image_id in tqdm(os.listdir(images_path)[:]):
                try:
                    image_path = images_path + "/" + image_id

                    prompt = json.load(open(prompts_path + "/" + image_id.replace(".png", ".json"), "r"))["bg_label"]
                    generated_image = generate_controlled_image(image_path, prompt, pipe=pipe)
                    save_path_dir = save_path + "/" + controlnet_model
                    if not os.path.exists(save_path_dir):
                        os.makedirs(save_path_dir)  
                    generated_image.save(save_path_dir + "/" + image_id)
                    # stripp.save(save_path_dir + "/" + image_id.split('.')[0] + '_strip.png')

                    overlaid_image = overlay_foreground(image_path, save_path_dir + "/" + image_id)
                    save_path_dir_overlay = save_path_dir + "_overlay"  
                    if not os.path.exists(save_path_dir_overlay):
                        os.makedirs(save_path_dir_overlay)
                    cv2.imwrite(save_path_dir_overlay + "/" + image_id, overlaid_image)
                except Exception as e:
                    print(e)
                    continue    
            # free memory
            del pipe
            torch.cuda.empty_cache()    
        except Exception as e:
            print(e)
            continue


























































