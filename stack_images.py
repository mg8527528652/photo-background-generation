from PIL import Image
import os
def stack_images_in_grid(image_paths, grid_dim):
    """
    Stack images in a grid layout.
    
    Args:
        image_paths: List of paths to images
        grid_dim: Tuple of (rows, columns) for the grid layout
        
    Returns:
        PIL.Image: Stacked image in grid format
    """
    rows, cols = grid_dim
    images = [Image.open(path) for path in image_paths]
    
    # Resize images to the same size
    min_width = min(image.size[0] for image in images)
    min_height = min(image.size[1] for image in images)
    images = [image.resize((min_width, min_height)) for image in images]
    
    # Create a new blank image for the grid
    grid_image = Image.new('RGB', (min_width * cols, min_height * rows))
    
    for index, image in enumerate(images):
        row = index // cols
        col = index % cols
        grid_image.paste(image, (col * min_width, row * min_height))
    
    return grid_image

# Example usage
root = '/root/photo-background-generation/sched'
folder_paths = os.listdir(root)
image_names = os.listdir(os.path.join(root, folder_paths[0], 'checkpoint-best_ckpt.pth-81730'))
os.makedirs(os.path.join(root, 'stacked_images'),exist_ok=True)
for sku in image_names:
    images_paths = []
    for folder_path in folder_paths:
        image_path = os.path.join(root, folder_path, 'checkpoint-best_ckpt.pth-81730', sku)
        images_paths.append(image_path)
    stacked_images= stack_images_in_grid(images_paths, (2, 4))
    stacked_images.save(os.path.join(root, 'stacked_images', sku))
        # print(image_path)
