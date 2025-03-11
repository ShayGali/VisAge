from tkinter import Frame
from .video_frame import VideoFrame

class AppWindow:
    def __init__(self, root):
        self.root = root
        self.root.geometry("800x600")
        
        # Create main frame
        self.frame = Frame(root)
        self.frame.pack(fill="both", expand=True)
        
        # Create video frame
        self.video_frame = VideoFrame(self.frame)
        self.video_frame.pack(fill="both", expand=True)