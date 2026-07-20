from tkinter import Tk
from tkinter.filedialog import askopenfilename

root = Tk()
root.withdraw()  # Hide the small tkinter window

file_path = askopenfilename(
    title="Select an Image",
    filetypes=[
        ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
        ("All Files", "*.*")
    ]
)

print(file_path)