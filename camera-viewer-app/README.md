# Camera Viewer App

This project is a simple application that displays the camera feed using Python's Tkinter library and OpenCV. It serves as a demonstration of integrating camera functionality into a GUI application.

## Project Structure

```
camera-viewer-app
├── src
│   ├── main.py          # Entry point of the application
│   ├── camera.py        # Handles camera operations
│   └── ui
│       ├── __init__.py  # Marks the ui directory as a package
│       ├── app_window.py # Defines the main application window
│       └── video_frame.py # Defines the video frame widget
├── config
│   └── settings.py      # Configuration settings for the application
├── requirements.txt      # Lists the dependencies required for the project
└── README.md             # Documentation for the project
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd camera-viewer-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command:
```
python src/main.py
```

Make sure your camera is connected and accessible. The application will open a window displaying the live camera feed.

## Dependencies

This project requires the following Python packages:
- OpenCV
- Tkinter

Make sure to install these packages using the `requirements.txt` file provided.

## License

This project is licensed under the MIT License - see the LICENSE file for details.