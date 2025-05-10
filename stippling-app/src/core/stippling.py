import numpy as np
import cv2
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
import time
from utils.file_handling import load_image, save_image

def grayscale_intensity(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    intensity = 255 - gray
    return intensity

def generate_weighted_points(intensity, n_points=1000):
    if intensity.max() > 0:
        normalized = intensity / intensity.max()
    else:
        normalized = intensity
    
    y_indices, x_indices = np.nonzero(normalized)
    probabilities = normalized[y_indices, x_indices]
    probabilities = probabilities / probabilities.sum()
    
    selected_indices = np.random.choice(
        len(y_indices), 
        size=min(n_points, len(y_indices)), 
        p=probabilities, 
        replace=False
    )
    
    points = np.column_stack([x_indices[selected_indices], y_indices[selected_indices]])
    return points

def lloyd_relaxation(points, intensity, iterations=10, progress_callback=None):
    """Lloyd relaxation with progress reporting."""
    height, width = intensity.shape
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    coords = np.vstack((x_coords.ravel(), y_coords.ravel())).T
    
    for i in range(iterations):
        vor = Voronoi(points)
        new_points = []
        for i, region_idx in enumerate(vor.point_region):
            region = vor.regions[region_idx]
            if -1 in region or len(region) == 0:
                new_points.append(points[i])
                continue
            
            polygon = [vor.vertices[v] for v in region]
            if not polygon:
                new_points.append(points[i])
                continue
            
            polygon_array = np.array(polygon)
            x_min, y_min = np.floor(polygon_array.min(axis=0)).astype(int)
            x_max, y_max = np.ceil(polygon_array.max(axis=0)).astype(int)
            
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(width-1, x_max)
            y_max = min(height-1, y_max)
            
            if x_min >= x_max or y_min >= y_max:
                new_points.append(points[i])
                continue
            
            x_range = np.arange(x_min, x_max+1)
            y_range = np.arange(y_min, y_max+1)
            xx, yy = np.meshgrid(x_range, y_range)
            pixels = np.vstack((xx.ravel(), yy.ravel())).T
            
            weights = np.array([intensity[y, x] if 0 <= y < height and 0 <= x < width else 0 
                              for x, y in pixels])
            
            if weights.sum() > 0:
                centroid = np.average(pixels, axis=0, weights=weights)
                new_points.append(centroid)
            else:
                new_points.append(points[i])
        
        points = np.array(new_points)
        
        # Report progress for each iteration
        if progress_callback:
            # Progress is from 0.3 to 0.9 during Lloyd relaxation
            progress = 0.3 + (i+1) / iterations * 0.6
            progress_callback(progress)
    
    return points

def detect_object_with_stippling(image_path, n_points=3000, lloyd_iterations=3, return_image=False, progress_callback=None):
    """
    Weighted Voronoi Stippling for object detection with progress reporting.
    
    Args:
        image_path (str): Path to the input image
        n_points (int): Number of stipple points
        lloyd_iterations (int): Number of Lloyd relaxation iterations
        return_image (bool): Whether to return the resulting image
        progress_callback (function): Callback to report progress (0-1)
        
    Returns:
        If return_image is True, returns the resulting image and points
        Otherwise, returns only the points
    """
    # Report initial progress
    if progress_callback:
        progress_callback(0.1)
    
    # Load the image using our utility function
    try:
        image = load_image(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
    except Exception as e:
        raise ValueError(f"Could not load image from {image_path}: {str(e)}")
    
    # Convert to grayscale and get intensity
    intensity = grayscale_intensity(image)
    
    # Report progress after loading
    if progress_callback:
        progress_callback(0.2)
    
    # Generate initial points
    points = generate_weighted_points(intensity, n_points)
    
    # Report progress after point generation
    if progress_callback:
        progress_callback(0.3)
    
    # Perform Lloyd relaxation with progress reporting
    relaxed_points = lloyd_relaxation(
        points, intensity, lloyd_iterations, 
        progress_callback=progress_callback if progress_callback else None
    )
    
    if return_image:
        # Create the output image
        output_image = np.ones_like(image) * 255  # White background
        
        # Draw black points
        for point in relaxed_points:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                cv2.circle(output_image, (x, y), 1, (0, 0, 0), -1)
        
        # Report completion
        if progress_callback:
            progress_callback(1.0)
            
        return output_image, relaxed_points
    else:
        # Report completion
        if progress_callback:
            progress_callback(1.0)
            
        return relaxed_points