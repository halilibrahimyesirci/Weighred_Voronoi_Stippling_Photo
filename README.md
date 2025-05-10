# Stippling App

## Overview
`Weighted Voronoi Stippling Project Explanation`
Hello! I'd like to walk you through the key components of our Weighted Voronoi Stippling application. This is a Python application that uses image processing techniques to create a stippled representation of an image, which can be useful for artistic effects and object detection.

`Main Application Structure`
The main application file (app.py) is quite simple - it configures the CustomTkinter appearance, creates the main window, and starts the application:

`Core Algorithm`
The core of our application is the Weighted Voronoi Stippling algorithm. Here's how it works:

`Image Processing:` We convert the image to grayscale and create an intensity map.
`Point Generation:` We generate points with higher density in darker areas of the image.
`Lloyd Relaxation:` This is an iterative process that evenly distributes points while respecting the density requirements.
`Boundary Constraints:` We've added constraints to keep points within the high-intensity regions (object boundaries).
The algorithm creates a stippled representation where each dot represents a portion of the image, with more dots in darker/more detailed areas.

`GUI Components`
Our GUI consists of several components:

`Main Window:` Coordinates all the other components and handles the main processing logic.
`Image Display:` Shows the original image, stippled result, and extracted object with transparency.
`Controls Frame:` Contains sliders for adjusting parameters like point count and iteration count.
`Progress Bar:` Shows the computational load and processing progress.
Object Extraction Feature
One of the more advanced features is the object extraction, which:

Uses the stippling points to identify the object region
Creates a mask from these points
Extracts the object from the original image
Makes the background transparent
This is particularly useful for isolating objects from their backgrounds for further use.

`Performance Considerations`
We've implemented several optimizations:

`Multithreading:` Processing happens in a background thread to keep the UI responsive
`Memory Management:` We're careful about how we store and process images to minimize RAM usage
`Efficient Algorithms:` We use vectorized operations with NumPy whenever possible

`File Handling`
Our application handles file operations with special care:

`Non-ASCII Characters:` We have utilities to handle filenames with special characters
Saving Options: Users can save both the stippled image and the extracted object
This makes the application more user-friendly, especially for international users.

## Features
- Load images and convert them to grayscale.
- Generate weighted points based on image intensity.
- Apply Lloyd relaxation for improved point distribution.
- Adjustable parameters for stippling through a graphical user interface.
- Progress bar to indicate processing status.

## Project Structure
```
stippling-app
├── src
│   ├── app.py                # Entry point of the application
│   ├── core
│   │   ├── __init__.py       # Core module initializer
│   │   └── stippling.py       # Main stippling logic
│   ├── gui
│   │   ├── __init__.py       # GUI module initializer
│   │   ├── components
│   │   │   ├── __init__.py    # Components submodule initializer
│   │   │   ├── controls_frame.py # UI controls for user input
│   │   │   ├── image_display.py   # Image display handling
│   │   │   └── progress_bar.py     # Progress bar implementation
│   │   └── main_window.py     # Main application window setup
│   └── utils
│       ├── __init__.py       # Utils module initializer
│       └── file_handling.py   # File handling utilities
├── assets
│   └── sample_images
│       └── readme.md         # Information about sample images
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd stippling-app
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Run the application:
   ```
   python src/app.py
   ```
2. Use the interface to load an image, adjust stippling parameters, and visualize the stippled output.

## NOTE:
If you wanted to test it with a simple code line there is `basic_code.py`.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
