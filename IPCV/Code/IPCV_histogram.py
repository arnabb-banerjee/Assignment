# IPCV_histogram.py
import numpy as np
from PIL import Image

class IPCV_histogram:
    
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
    
    def compute_histogram(self, image):
        """Compute normalized histogram of the image"""
        hist = np.zeros(256)
        for pixel in image.flatten():
            hist[int(pixel)] += 1
        hist /= image.size  # Normalize
        return hist
    
    def histogram_equalization(self, image):
        """Perform histogram equalization"""
        # Compute histogram and CDF
        hist = self.compute_histogram(image)
        cdf = np.cumsum(hist)
        
        # Normalize CDF to 0-255 range
        cdf_normalized = (cdf * 255 / cdf[-1]).astype(np.uint8)
        
        # Map original values to equalized values
        equalized = np.zeros_like(image)
        for i in range(256):
            equalized[image == i] = cdf_normalized[i]
            
        return self.normalize(equalized)
    
    def contrast_stretching(self, image, low=2, high=98):
        """Perform contrast stretching using percentile values"""
        # Convert percentiles to pixel values
        plow = np.percentile(image, low)
        phigh = np.percentile(image, high)
        
        # Apply contrast stretching
        stretched = np.clip((image - plow) * (255.0 / (phigh - plow)), 0, 255)
        return stretched.astype(np.uint8)