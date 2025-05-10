import customtkinter as ctk

class ProgressBar(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        
        # Configure the frame
        self.configure(fg_color=("gray85", "gray15"))
        
        # Create progress bar and load label
        self.progress_label = ctk.CTkLabel(self, text="Current Load:")
        self.progress_label.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="w")
        
        self.progress_bar = ctk.CTkProgressBar(self, orientation="horizontal")
        self.progress_bar.grid(row=0, column=1, padx=5, pady=10, sticky="ew")
        
        self.load_value_label = ctk.CTkLabel(self, text="0%")
        self.load_value_label.grid(row=0, column=2, padx=(5, 10), pady=10, sticky="e")
        
        # Configure column weights
        self.grid_columnconfigure(0, weight=0)  # Label doesn't need to expand
        self.grid_columnconfigure(1, weight=1)  # Progress bar should expand
        self.grid_columnconfigure(2, weight=0)  # Value label doesn't need to expand
        
        # Set initial progress value
        self.progress_bar.set(0)
    
    def update_load(self, value):
        """
        Update the progress bar value
        
        Args:
            value (float): Value between 0 and 1
        """
        # Ensure value is between 0 and 1
        value = max(0, min(1, value))
        
        # Update progress bar
        self.progress_bar.set(value)
        
        # Update label
        self.load_value_label.configure(text=f"{int(value * 100)}%")