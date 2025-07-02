# IPCV_point_processing.py
import numpy as np
from PIL import Image

class IPCV_point_processing:
    
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
    
    def negative(self, image):
        """Compute negative of the image"""
        return self.normalize(255 - image)
    
    def threshold(self, image, value=128):
        """Apply threshold to the image"""
        thresholded = np.zeros_like(image)
        thresholded[image > value] = 255
        return thresholded.astype(np.uint8)
    
    def power_law_transform(self, image, gamma=1.0, c=1.0):
        """Apply power-law (gamma) transform to the image"""
        # Normalize to 0-1 range first
        normalized = image / 255.0
        # Apply gamma correction
        transformed = c * np.power(normalized, gamma)
        # Scale back to 0-255 range
        return self.normalize(transformed * 255)