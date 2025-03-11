from tkinter import Tk
from ui.app_window import AppWindow

def main():
    root = Tk()
    root.title("Camera Viewer")
    app = AppWindow(root)
    root.mainloop()

if __name__ == "__main__":
    main()