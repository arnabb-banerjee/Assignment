# IPCV_spatial_filter.py
import numpy as np
from PIL import Image

class IPCV_spatial_filter:
    
    def load_image(self, image_path):
        """Load an image and convert to grayscale numpy array"""
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        return np.array(img, dtype=np.float32)
    
    def normalize(self, arr):
        """Normalize array to 0-255 range"""
        arr = arr.astype(np.float32)
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max == arr_min:
            return np.zeros_like(arr, dtype=np.uint8)
        return ((arr - arr_min) * (255 / (arr_max - arr_min))).clip(0, 255).astype(np.uint8)
    
    def average_filter(self, image, size=3):
        """Apply average filter to the image"""
        pad = size // 2
        filtered = np.zeros_like(image)
        rows, cols = image.shape
        
        # Pad the image
        padded = np.pad(image, pad, mode='reflect')
        
        for i in range(rows):
            for j in range(cols):
                region = padded[i:i+size, j:j+size]
                filtered[i,j] = np.mean(region)
                
        return self.normalize(filtered)
    
    def median_filter(self, image, size=3):
        """Apply median filter to the image"""
        pad = size // 2
        filtered = np.zeros_like(image)
        rows, cols = image.shape
        
        # Pad the image
        padded = np.pad(image, pad, mode='reflect')
        
        for i in range(rows):
            for j in range(cols):
                region = padded[i:i+size, j:j+size]
                filtered[i,j] = np.median(region)
                
        return self.normalize(filtered)