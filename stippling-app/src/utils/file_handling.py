import os
import cv2
import numpy as np
from PIL import Image

def load_image(image_path):
    """
    Load an image with support for non-ASCII characters in the path.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Loaded image
    """
    try:
        # First try with OpenCV directly
        img = cv2.imread(image_path)
        if img is not None:
            return img
            
        # If OpenCV fails, try with PIL and convert to OpenCV format
        pil_image = Image.open(image_path)
        # Convert to RGB mode to ensure compatibility
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        # Convert PIL image to numpy array
        img = np.array(pil_image)
        # Convert RGB to BGR (OpenCV format)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        raise ValueError(f"Failed to load image from {image_path}: {str(e)}")

def save_image(image, output_path):
    """
    Save an image with support for non-ASCII characters in the path.
    
    Args:
        image (numpy.ndarray): Image to save
        output_path (str): Path where to save the image
    """
    try:
        # Convert BGR to RGB for PIL
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        # Convert to PIL Image and save
        pil_image = Image.fromarray(image_rgb)
        pil_image.save(output_path)
    except Exception as e:
        raise ValueError(f"Failed to save image to {output_path}: {str(e)}")

def get_image_extension(image_path):
    """Return the file extension of the image."""
    return image_path.split('.')[-1].lower()