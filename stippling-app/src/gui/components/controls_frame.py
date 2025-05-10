import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import os

class ControlsFrame(ctk.CTkFrame):
    def __init__(self, master, update_callback):
        super().__init__(master)
        self.update_callback = update_callback
        
        # Configure the frame
        self.configure(width=300, fg_color=("gray90", "gray13"))
        
        # Label
        self.title_label = ctk.CTkLabel(self, text="Stippling Controls", font=ctk.CTkFont(size=16, weight="bold"))
        self.title_label.pack(pady=(20, 10), padx=10)
        
        # File selection
        self.file_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.file_frame.pack(fill="x", padx=10, pady=10)
        
        self.file_button = ctk.CTkButton(self.file_frame, text="Select Image", command=self.select_file)
        self.file_button.pack(side="left", padx=5)
        
        self.file_label = ctk.CTkLabel(self.file_frame, text="No file selected")
        self.file_label.pack(side="left", padx=5, fill="x")
        
        # Number of points slider
        self.points_label = ctk.CTkLabel(self, text=f"Number of Points: 3000")
        self.points_label.pack(pady=(20, 0), padx=10, anchor="w")
        
        self.points_slider = ctk.CTkSlider(self, from_=100, to=10000, number_of_steps=99)
        self.points_slider.set(3000)
        self.points_slider.configure(command=self.update_points)
        self.points_slider.pack(pady=(0, 10), padx=20, fill="x")
        
        # Lloyd iterations slider
        self.lloyd_label = ctk.CTkLabel(self, text=f"Lloyd Iterations: 3")
        self.lloyd_label.pack(pady=(10, 0), padx=10, anchor="w")
        
        self.lloyd_slider = ctk.CTkSlider(self, from_=1, to=10, number_of_steps=9)
        self.lloyd_slider.set(3)
        self.lloyd_slider.configure(command=self.update_lloyd_iterations)
        self.lloyd_slider.pack(pady=(0, 10), padx=20, fill="x")
        
        # Process button
        self.process_button = ctk.CTkButton(self, text="Process Image", command=self.process_image)
        self.process_button.pack(pady=20, padx=20)
    
    def select_file(self):
        filetypes = (
            ('Image files', '*.png *.jpg *.jpeg *.bmp *.tiff'),
            ('All files', '*.*')
        )
        
        filename = filedialog.askopenfilename(
            title='Open an image',
            initialdir='/',
            filetypes=filetypes
        )
        
        if filename:
            # Display only the filename, not the full path
            display_name = os.path.basename(filename)
            self.file_label.configure(text=display_name)
            
            # Consider copying the file to a temp location with a simple name
            # if the filename contains non-ASCII characters
            if any(ord(c) > 127 for c in filename):
                import tempfile
                import shutil
                
                # Create a temporary file with a simple ASCII name
                temp_dir = tempfile.gettempdir()
                ext = os.path.splitext(filename)[1]
                temp_filename = os.path.join(temp_dir, f"stippling_temp{ext}")
                
                # Copy the original file to the temp location
                shutil.copy2(filename, temp_filename)
                
                # Use the temp file instead
                print(f"Using temporary file: {temp_filename} (original: {filename})")
                filename = temp_filename
            
            # Pass the full path to the MainWindow
            self.update_callback('image_path', filename)
    
    def update_points(self, value):
        points = int(value)
        self.points_label.configure(text=f"Number of Points: {points}")
        self.update_callback('n_points', points)
    
    def update_lloyd_iterations(self, value):
        iterations = int(value)
        self.lloyd_label.configure(text=f"Lloyd Iterations: {iterations}")
        self.update_callback('lloyd_iterations', iterations)
    
    def process_image(self):
        # Call a method in the parent window to start processing with current settings
        if hasattr(self.master, 'update_stippling'):
            self.master.update_stippling()