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

def constrained_lloyd_relaxation(points, intensity, iterations=10, intensity_threshold=10, progress_callback=None):
    """
    Constrained Lloyd relaxation algorithm that keeps points only in high-intensity areas.
    
    Args:
        points: Initial points
        intensity: Intensity map
        iterations: Number of iterations
        intensity_threshold: Minimum intensity threshold for valid regions
        progress_callback: Callback function for progress updates
        
    Returns:
        Adjusted points
    """
    height, width = intensity.shape
    
    # Create intensity threshold binary mask
    valid_region = intensity > intensity_threshold
    
    for i in range(iterations):
        # Create Voronoi diagram
        # Use extended area to avoid boundary problems
        padded_points = np.vstack([points, 
                                  [[-width/2, -height/2], [width*1.5, -height/2], 
                                   [-width/2, height*1.5], [width*1.5, height*1.5]]])
        vor = Voronoi(padded_points)
        
        # Her bölge için ağırlıklı merkez hesapla
        new_points = []
        for j, region_idx in enumerate(vor.point_region[:len(points)]):  # Sadece orijinal noktaları işle
            region = vor.regions[region_idx]
            if -1 in region or len(region) == 0:
                # Invalid region, keep point but constrain to valid area
                new_point = constrain_point_to_valid_region(points[j], valid_region, intensity)
                new_points.append(new_point)
                continue
                
            # Get region polygon
            polygon = [vor.vertices[v] for v in region]
            if not polygon:
                new_point = constrain_point_to_valid_region(points[j], valid_region, intensity)
                new_points.append(new_point)
                continue
                
            # Constrain polygon
            polygon_array = np.array(polygon)
            x_min, y_min = np.floor(polygon_array.min(axis=0)).astype(int)
            x_max, y_max = np.ceil(polygon_array.max(axis=0)).astype(int)
            
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(width-1, x_max)
            y_max = min(height-1, y_max)
            
            if x_min >= x_max or y_min >= y_max:
                new_point = constrain_point_to_valid_region(points[j], valid_region, intensity)
                new_points.append(new_point)
                continue
            
            # Check pixels within polygon
            x_range = np.arange(x_min, x_max+1)
            y_range = np.arange(y_min, y_max+1)
            xx, yy = np.meshgrid(x_range, y_range)
            pixels = np.vstack((xx.ravel(), yy.ravel())).T
            
            # Calculate centroid - only in valid regions
            weights = np.array([
                intensity[y, x] if 0 <= y < height and 0 <= x < width and valid_region[y, x] else 0 
                for x, y in pixels
            ])
            
            if weights.sum() > 0:
                centroid = np.average(pixels, axis=0, weights=weights)
                new_points.append(centroid)
            else:
                # If no weight in valid region, move point to nearest valid region
                new_point = constrain_point_to_valid_region(points[j], valid_region, intensity)
                new_points.append(new_point)
        
        points = np.array(new_points)
        
        # Progress reporting
        if progress_callback:
            # Progress goes from 0.3 to 0.9 during Lloyd relaxation
            progress = 0.3 + (i+1) / iterations * 0.6
            progress_callback(progress)
    
    return points

def constrain_point_to_valid_region(point, valid_region, intensity):
    """
    Moves a point to a valid region (high intensity area).
    
    Args:
        point: Point to move [x, y]
        valid_region: Boolean mask of valid regions
        intensity: Intensity map
        
    Returns:
        Point moved to valid region
    """
    x, y = int(point[0]), int(point[1])
    height, width = valid_region.shape
    
    # Nokta zaten geçerli bölgedeyse, olduğu gibi bırak
    if 0 <= x < width and 0 <= y < height and valid_region[y, x]:
        return point
    
    # En yakın geçerli bölgeyi bul
    y_coords, x_coords = np.where(valid_region)
    
    if len(y_coords) == 0:  # Hiç geçerli bölge yoksa
        return point
    
    # Tüm geçerli noktalara olan uzaklıkları hesapla
    distances = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
    
    # En kısa mesafedeki noktayı bul
    closest_idx = np.argmin(distances)
    closest_x, closest_y = x_coords[closest_idx], y_coords[closest_idx]
    
    # Yeni nokta
    return np.array([closest_x, closest_y])

def extract_object_from_stippling(original_image, stipple_points, dilation_size=5):
    """
    Extracts object from original image using stipple points and makes background transparent.
    
    Args:
        original_image: Original input image
        stipple_points: Points generated by stippling algorithm
        dilation_size: Size of mask dilation
        
    Returns:
        RGBA image with transparent background
    """
    # Noktalardan maske oluştur
    mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
    
    # Noktaları beyaz noktalar olarak çiz
    for point in stipple_points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
            mask[y, x] = 255
    
    # Maskeyi genişlet ve yumuşat
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)
    blurred_mask = cv2.GaussianBlur(dilated_mask, (21, 21), 0)
    
    # Maskeyi normalize et
    _, mask_binary = cv2.threshold(blurred_mask, 50, 255, cv2.THRESH_BINARY)
    
    # Maske üzerinde contourları bul
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # En büyük contour'u al - bu muhtemelen nesnenin kendisidir
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Nesneyi içeren maske oluştur
        refined_mask = np.zeros_like(mask_binary)
        cv2.drawContours(refined_mask, [largest_contour], 0, 255, -1)
        
        # Maskeyi yumuşat
        refined_mask = cv2.GaussianBlur(refined_mask, (11, 11), 0)
        _, refined_mask = cv2.threshold(refined_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Alpha kanalı olarak maskeyi kullan
        b, g, r = cv2.split(original_image)
        alpha = refined_mask.copy()
        
        # RGBA formatında çıktı görüntüsü oluştur
        rgba_image = cv2.merge((r, g, b, alpha))
        
        return rgba_image
    else:
        # Contour bulunamadıysa, tüm görüntüyü döndür
        b, g, r = cv2.split(original_image)
        alpha = np.ones_like(mask) * 255
        rgba_image = cv2.merge((r, g, b, alpha))
        return rgba_image

def detect_object_with_stippling(image_path, n_points=3000, lloyd_iterations=3, return_image=False, 
                                extract_object=False, progress_callback=None):
    """
    Weighted Voronoi Stippling for object detection with progress reporting.
    
    Args:
        image_path (str): Path to the input image
        n_points (int): Number of stipple points
        lloyd_iterations (int): Number of Lloyd relaxation iterations
        return_image (bool): Whether to return the resulting image
        extract_object (bool): Whether to extract the object from the original image
        progress_callback (function): Callback to report progress (0-1)
        
    Returns:
        If return_image is True, returns the resulting image, points, and (optionally) extracted object
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
    
    result = [relaxed_points]
    
    if return_image:
        # Create the output image
        output_image = np.ones_like(image) * 255  # White background
        
        # Draw black points
        for point in relaxed_points:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                cv2.circle(output_image, (x, y), 1, (0, 0, 0), -1)
        
        result.insert(0, output_image)
        
        # Extract object if requested
        if extract_object:
            extracted_image = extract_object_from_stippling(image, relaxed_points)
            result.append(extracted_image)
    
    # Report completion
    if progress_callback:
        progress_callback(1.0)
        
    return tuple(result)