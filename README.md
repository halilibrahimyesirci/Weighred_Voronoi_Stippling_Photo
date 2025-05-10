# Stippling App

## Overview
The Stippling App is a Python application that utilizes image processing techniques to create stippled representations of images. It features a user-friendly interface built with CustomTkinter, allowing users to adjust stippling parameters and visualize the results in real-time.

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

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.