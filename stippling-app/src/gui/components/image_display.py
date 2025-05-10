import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np
from tkinter import filedialog
import os
from utils.file_handling import load_image, save_image

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
        self.tab_view.add("Extracted Object")  # New tab for extracted object
        
        # Create image labels
        self.original_image_label = ctk.CTkLabel(self.tab_view.tab("Original Image"), text="No image loaded")
        self.original_image_label.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Stippled image with save button
        self.stippled_frame = ctk.CTkFrame(self.tab_view.tab("Stippled Image"), fg_color="transparent")
        self.stippled_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.stippled_image_label = ctk.CTkLabel(self.stippled_frame, text="No processed image yet")
        self.stippled_image_label.pack(fill="both", expand=True, padx=10, pady=(10, 5))
        
        # Add save button for stippled image
        self.save_stippled_button = ctk.CTkButton(
            self.stippled_frame, 
            text="Save Stippled Image", 
            command=self.save_stippled_image,
            state="disabled"
        )
        self.save_stippled_button.pack(pady=(0, 10))
        
        # Extracted object with save button
        self.extracted_frame = ctk.CTkFrame(self.tab_view.tab("Extracted Object"), fg_color="transparent")
        self.extracted_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.extracted_image_label = ctk.CTkLabel(self.extracted_frame, text="No extracted object yet")
        self.extracted_image_label.pack(fill="both", expand=True, padx=10, pady=(10, 5))
        
        # Add save button for extracted image
        self.save_extracted_button = ctk.CTkButton(
            self.extracted_frame, 
            text="Save Extracted Object", 
            command=self.save_extracted_image,
            state="disabled"
        )
        self.save_extracted_button.pack(pady=(0, 10))
        
        # Keep references to the images to prevent garbage collection
        self.original_photo = None
        self.stippled_photo = None
        self.extracted_photo = None
        
        # Store actual image data
        self.original_image = None
        self.stippled_image = None
        self.extracted_image = None
    
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
            # Store the original stippled image
            self.stippled_image = image.copy()
            
            # Convert to RGB if it's not already
            if len(image.shape) == 2 or image.shape[2] == 1:
                display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3 and image.dtype == np.uint8:
                # If BGR (from OpenCV), convert to RGB
                display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                display_image = image.copy()
            
            # Resize to fit the frame while maintaining aspect ratio
            resized_image = self._resize_image_to_fit(display_image)
            
            # Convert to PhotoImage
            pil_image = Image.fromarray(resized_image)
            self.stippled_photo = ImageTk.PhotoImage(pil_image)
            
            # Update the label
            self.stippled_image_label.configure(image=self.stippled_photo, text="")
            
            # Enable save button
            self.save_stippled_button.configure(state="normal")
            
            # Switch to the stippled image tab
            self.tab_view.set("Stippled Image")
            
        except Exception as e:
            print(f"Error displaying stippled image: {e}")
            self.stippled_image_label.configure(text=f"Error displaying stippled image: {e}", image=None)
    
    def show_extracted_image(self, image):
        """Display the extracted object image"""
        try:
            # Store the original extracted image
            self.extracted_image = image.copy()
            
            # Resize to fit the frame while maintaining aspect ratio
            resized_image = self._resize_image_to_fit(image)
            
            # Convert to PhotoImage
            pil_image = Image.fromarray(resized_image)
            self.extracted_photo = ImageTk.PhotoImage(pil_image)
            
            # Update the label
            self.extracted_image_label.configure(image=self.extracted_photo, text="")
            
            # Enable save button
            self.save_extracted_button.configure(state="normal")
            
        except Exception as e:
            print(f"Error displaying extracted image: {e}")
            self.extracted_image_label.configure(text=f"Error displaying extracted image: {e}", image=None)
    
    def save_stippled_image(self):
        """Save the stippled image to disk"""
        if self.stippled_image is None:
            print("No stippled image to save")
            return
            
        filetypes = (
            ('PNG Image', '*.png'),
            ('JPEG Image', '*.jpg'),
            ('All files', '*.*')
        )
        
        filename = filedialog.asksaveasfilename(
            title='Save Stippled Image',
            filetypes=filetypes,
            defaultextension=".png"
        )
        
        if filename:
            try:
                save_image(self.stippled_image, filename)
                print(f"Stippled image saved to {filename}")
            except Exception as e:
                print(f"Error saving stippled image: {e}")
    
    def save_extracted_image(self):
        """Save the extracted object image to disk"""
        if self.extracted_image is None:
            print("No extracted object image to save")
            return
            
        filetypes = (
            ('PNG Image', '*.png'),
            ('JPEG Image', '*.jpg'),
            ('All files', '*.*')
        )
        
        filename = filedialog.asksaveasfilename(
            title='Save Extracted Object',
            filetypes=filetypes,
            defaultextension=".png"
        )
        
        if filename:
            try:
                # For transparent images, always save as PNG
                if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
                    print("Warning: Transparent image should be saved as PNG. Converting...")
                    filename = os.path.splitext(filename)[0] + ".png"
                
                save_image(self.extracted_image, filename)
                print(f"Extracted object image saved to {filename}")
            except Exception as e:
                print(f"Error saving extracted object image: {e}")
    
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