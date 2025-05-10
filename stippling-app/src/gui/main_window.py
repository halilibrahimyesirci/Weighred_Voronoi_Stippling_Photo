import customtkinter as ctk
from gui.components.controls_frame import ControlsFrame
from gui.components.image_display import ImageDisplay
from gui.components.progress_bar import ProgressBar
from core.stippling import detect_object_with_stippling
import threading
import cv2
from PIL import Image, ImageTk
import numpy as np
import time
import os

class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Weighted Voronoi Stippling")
        self.geometry("1200x800")
        self.minsize(900, 600)
        
        # Initialize settings and state variables
        self.settings = {
            'n_points': 3000,
            'lloyd_iterations': 3
        }
        self.image_path = None
        self.processing = False
        
        # Create main layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        
        # Create image display area
        self.image_display = ImageDisplay(self)
        self.image_display.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Create controls frame with update callback
        self.controls_frame = ControlsFrame(self, self.update_setting)
        self.controls_frame.grid(row=0, column=1, padx=10, pady=10, sticky="ns")
        
        # Create progress bar
        self.progress_bar = ProgressBar(self)
        self.progress_bar.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
    
    def update_setting(self, setting_name, value):
        """
        Callback function to update a specific setting
        
        Args:
            setting_name (str): Name of the setting to update
            value: New value for the setting
        """
        # Update the setting
        self.settings[setting_name] = value
        
        # Calculate load based on settings and update progress bar
        load = self._calculate_load()
        self.progress_bar.update_load(load)
        
        print(f"Updated {setting_name} to {value}. Current settings: {self.settings}")
        
        if setting_name == 'image_path':
            self.image_path = value
            # Display the selected image in the image display area
            self.image_display.show_original_image(value)
    
    def _calculate_load(self):
        """Calculate the computational load based on current settings"""
        # Simple load calculation formula
        n_points = self.settings.get('n_points', 3000)
        lloyd_iterations = self.settings.get('lloyd_iterations', 3)
        
        # Load increases with more points and more iterations
        # Normalize to a 0-1 scale
        load = (n_points / 10000) * (lloyd_iterations / 10)
        return min(1.0, max(0.0, load))  # Clamp between 0 and 1
    
    def update_stippling(self):
        """
        Execute the stippling process with current settings
        """
        if not self.image_path or self.processing:
            return
            
        if not os.path.exists(self.image_path):
            print(f"Error: Image file not found at {self.image_path}")
            return
            
        self.processing = True
        self.controls_frame.process_button.configure(state="disabled", text="Processing...")
        self.progress_bar.update_load(0.1)  # Show initial progress
        
        # Run the processing in a separate thread to avoid freezing the UI
        threading.Thread(target=self._process_image_thread, daemon=True).start()
    
    def _process_image_thread(self):
        """Background thread to process the image"""
        try:
            # Get settings
            n_points = self.settings.get('n_points', 3000)
            lloyd_iterations = self.settings.get('lloyd_iterations', 3)
            
            # Update progress
            self.after(100, lambda: self.progress_bar.update_load(0.2))
            
            # Call the stippling function
            start_time = time.time()
            
            # Call the updated stippling algorithm with object extraction
            result = detect_object_with_stippling(
                self.image_path,
                n_points=n_points,
                lloyd_iterations=lloyd_iterations,
                return_image=True,
                extract_object=True,  # Enable object extraction
                progress_callback=self._update_progress
            )
            
            # Unpack the results - should be (stippled_image, points, extracted_image)
            if len(result) == 3:
                stippled_image, points, extracted_image = result
            else:
                stippled_image, points = result
                extracted_image = None
            
            processing_time = time.time() - start_time
            print(f"Processing completed in {processing_time:.2f} seconds")
            
            # Update the UI with the results
            self.after(100, lambda: self._update_ui_with_result(stippled_image, extracted_image))
            
        except Exception as e:
            error_message = str(e)
            print(f"Error processing image: {error_message}")
            self.after(100, lambda error=error_message: self._handle_processing_error(error))
        finally:
            # Reset processing state
            self.after(100, lambda: self._reset_processing_state())
    
    def _update_progress(self, progress_value):
        """Update the progress bar from the processing thread"""
        self.after(10, lambda: self.progress_bar.update_load(progress_value))
    
    def _update_ui_with_result(self, stippled_image, extracted_image=None):
        """Update the UI with the processed images"""
        # Show the stippled image
        self.image_display.show_stippled_image(stippled_image)
        
        # Show the extracted object if available
        if extracted_image is not None:
            self.image_display.show_extracted_image(extracted_image)
        
        # Update progress bar to completion
        self.progress_bar.update_load(1.0)
    
    def _handle_processing_error(self, error_message):
        """Handle errors during processing"""
        print(f"Error: {error_message}")
        # Here you could show an error dialog
    
    def _reset_processing_state(self):
        """Reset the UI state after processing"""
        self.processing = False
        self.controls_frame.process_button.configure(state="normal", text="Process Image")