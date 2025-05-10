import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np
from utils.file_handling import load_image

class ImageDisplay(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        
        # Configure frame
        self.configure(fg_color=("gray80", "gray20"))
        
        # Create layout with tabs for original and stippled image
        self.tab_view = ctk.CTkTabview(self)
        self.tab_view.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create tabs
        self.tab_view.add("Original Image")
        self.tab_view.add("Stippled Image")
        
        # Create image labels
        self.original_image_label = ctk.CTkLabel(self.tab_view.tab("Original Image"), text="No image loaded")
        self.original_image_label.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.stippled_image_label = ctk.CTkLabel(self.tab_view.tab("Stippled Image"), text="No processed image yet")
        self.stippled_image_label.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Keep references to the images to prevent garbage collection
        self.original_photo = None
        self.stippled_photo = None
    
    def show_original_image(self, image_path):
        """Display the original image"""
        try:
            # Load the image using our utility function
            image = load_image(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to fit the frame while maintaining aspect ratio
            resized_image = self._resize_image_to_fit(image)
            
            # Convert to PhotoImage
            pil_image = Image.fromarray(resized_image)
            self.original_photo = ImageTk.PhotoImage(pil_image)
            
            # Update the label
            self.original_image_label.configure(image=self.original_photo, text="")
            
            # Switch to the original image tab
            self.tab_view.set("Original Image")
            
        except Exception as e:
            print(f"Error loading image: {e}")
            self.original_image_label.configure(text=f"Error loading image: {e}", image=None)
    
    def show_stippled_image(self, image):
        """Display the stippled image"""
        try:
            # Convert to RGB if it's not already
            if len(image.shape) == 2 or image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3 and image.dtype == np.uint8:
                # If BGR (from OpenCV), convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to fit the frame while maintaining aspect ratio
            resized_image = self._resize_image_to_fit(image)
            
            # Convert to PhotoImage
            pil_image = Image.fromarray(resized_image)
            self.stippled_photo = ImageTk.PhotoImage(pil_image)
            
            # Update the label
            self.stippled_image_label.configure(image=self.stippled_photo, text="")
            
            # Switch to the stippled image tab
            self.tab_view.set("Stippled Image")
            
        except Exception as e:
            print(f"Error displaying stippled image: {e}")
            self.stippled_image_label.configure(text=f"Error displaying stippled image: {e}", image=None)
    
    def _resize_image_to_fit(self, image, max_width=800, max_height=600):
        """Resize image to fit within the specified dimensions while maintaining aspect ratio"""
        height, width = image.shape[:2]
        
        # Calculate aspect ratios
        width_ratio = max_width / width
        height_ratio = max_height / height
        
        # Use the smaller ratio to ensure the image fits within both dimensions
        ratio = min(width_ratio, height_ratio)
        
        # Calculate new dimensions
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return resized