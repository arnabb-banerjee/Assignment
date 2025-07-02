# IPCV_image_restoration.py
import numpy as np
from PIL import Image
import random

class IPCV_image_restoration:
    
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
    
    def add_salt_pepper_noise(self, image, prob=0.05):
        """Add salt and pepper noise to the image"""
        noisy = np.copy(image)
        rows, cols = image.shape
        num_pixels = rows * cols
        
        # Add salt noise (white pixels)
        salt = int(num_pixels * prob / 2)
        coords = [np.random.randint(0, i-1, salt) for i in image.shape]
        noisy[coords[0], coords[1]] = 255
        
        # Add pepper noise (black pixels)
        pepper = int(num_pixels * prob / 2)
        coords = [np.random.randint(0, i-1, pepper) for i in image.shape]
        noisy[coords[0], coords[1]] = 0
        
        return noisy.astype(np.uint8)
    
    def adaptive_median_filter(self, image, max_size=7):
        """Apply adaptive median filter to the image"""
        filtered = np.copy(image)
        rows, cols = image.shape
        
        for i in range(rows):
            for j in range(cols):
                window_size = 3
                done = False
                
                while window_size <= max_size and not done:
                    pad = window_size // 2
                    
                    # Get the window region with boundary handling
                    i_min = max(0, i - pad)
                    i_max = min(rows, i + pad + 1)
                    j_min = max(0, j - pad)
                    j_max = min(cols, j + pad + 1)
                    
                    window = image[i_min:i_max, j_min:j_max]
                    z_min = np.min(window)
                    z_max = np.max(window)
                    z_med = np.median(window)
                    z_xy = image[i,j]
                    
                    # Level A
                    a1 = z_med - z_min
                    a2 = z_med - z_max
                    
                    if a1 > 0 and a2 < 0:
                        # Level B
                        b1 = z_xy - z_min
                        b2 = z_xy - z_max
                        
                        if b1 > 0 and b2 < 0:
                            filtered[i,j] = z_xy
                        else:
                            filtered[i,j] = z_med
                        done = True
                    else:
                        window_size += 2
                        
                        if window_size > max_size:
                            filtered[i,j] = z_med
                            done = True
        
        return self.normalize(filtered)