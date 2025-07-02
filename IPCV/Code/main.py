# main.py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from IPCV_edge_detect import IPCV_edge_detect
from IPCV_histogram import IPCV_histogram
from IPCV_spatial_filter import IPCV_spatial_filter
from IPCV_point_processing import IPCV_point_processing
from IPCV_image_restoration import IPCV_image_restoration

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

def display_results(original, results, titles):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, len(results)+1, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    for i, (result, title) in enumerate(zip(results, titles), 2):
        plt.subplot(1, len(results)+1, i)
        plt.imshow(result, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    print("Select an image processing operation:")
    print("1. Edge Detection")
    print("2. Histogram Processing")
    print("3. Spatial Filtering")
    print("4. Point Processing")
    print("5. Image Restoration")
    choice = input("Enter your choice (1-5): ")
    
    image_path = select_image()
    if not image_path:
        print("No image selected. Exiting.")
        return
    
    original_image = Image.open(image_path).convert('L')  # Convert to grayscale
    original_array = np.array(original_image, dtype=np.float32)
    
    if choice == '1':
        processor = IPCV_edge_detect()
        simple_edges = processor.simple_edge_detection(original_array, threshold=30)
        sobel_edges = processor.sobel_edge_detection(original_array, threshold=100)
        display_results(original_array, [simple_edges, sobel_edges], 
                       ["Simple Edge Detection", "Sobel Edge Detection"])
        
    elif choice == '2':
        processor = IPCV_histogram()
        equalized = processor.histogram_equalization(original_array)
        stretched = processor.contrast_stretching(original_array)
        display_results(original_array, [equalized, stretched], 
                       ["Histogram Equalization", "Contrast Stretching"])
        
    elif choice == '3':
        processor = IPCV_spatial_filter()
        avg_filtered = processor.average_filter(original_array, size=3)
        median_filtered = processor.median_filter(original_array, size=3)
        display_results(original_array, [avg_filtered, median_filtered], 
                       ["Average Filter", "Median Filter"])
        
    elif choice == '4':
        processor = IPCV_point_processing()
        negative = processor.negative(original_array)
        thresholded = processor.threshold(original_array, value=128)
        power_law = processor.power_law_transform(original_array, gamma=0.5)
        display_results(original_array, [negative, thresholded, power_law], 
                       ["Negative Image", "Thresholded", "Power Law (γ=0.5)"])
        
    elif choice == '5':
        processor = IPCV_image_restoration()
        # Add some salt & pepper noise to demonstrate restoration
        noisy = processor.add_salt_pepper_noise(original_array, prob=0.05)
        restored = processor.adaptive_median_filter(noisy, max_size=7)
        display_results(original_array, [noisy, restored], 
                       ["Noisy Image", "Restored Image"])
        
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()