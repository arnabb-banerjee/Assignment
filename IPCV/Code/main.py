import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from IPCV_edge_detect import IPCV_edge_detect
from tkinter import Tk, filedialog

# (Include all your function definitions here - load_image, normalize, simple_edge_detection, sobel_edge_detection)

def main():
    edgedetect = IPCV_edge_detect()
    # Step 1: Load an image
    #image_path = "your_image.jpg"  # Replace with your image path
    image_path = select_image()
    original_image = edgedetect.load_image(image_path)
    
    # Step 2: Apply simple edge detection
    simple_edges = edgedetect.simple_edge_detection(original_image, threshold=30)
    
    # Step 3: Apply Sobel edge detection
    sobel_edges = edgedetect.sobel_edge_detection(original_image, threshold=100)
    
    # Step 4: Display results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(simple_edges, cmap='gray')
    plt.title("Simple Edge Detection")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(sobel_edges, cmap='gray')
    plt.title("Sobel Edge Detection")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Step 5: Save results (optional)
    Image.fromarray(simple_edges).save("simple_edges.jpg")
    Image.fromarray(sobel_edges).save("sobel_edges.jpg")


def select_image():
    """Open a file dialog to select an image"""
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif")]
    )
    root.destroy()
    return file_path

if __name__ == "__main__":
    main()