import numpy as np
from PIL import Image

class IPCV_edge_detect:

    def load_image(self, image_path):
        """Load an image and convert to grayscale numpy array"""
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        return np.array(img, dtype=np.float32)

    def normalize(self, arr):
        """intensity refers to the brightness or lightness of a pixel in a grayscale image.
            For an 8-bit image, intensity ranges from:
                1. 0 (Pure Black) → Minimum brightness.
                2. 255 (Pure White) → Maximum brightness.
                3. Intermediate values (e.g., 128) → Shades of gray.

        If working with a color image (RGB), intensity can be computed as a weighted average of the red (R), green (G), and blue (B) channels:
        I = 0.299 × R + 0.587 × G + 0.114 × B 
        (This is the standard luminance formula used in PIL.Image.convert('L').)

        Normalize
        Crucial part of image processing, when we need to scale pixel values to a standard range (e.g., 0–255 for 8-bit grayscale images)
        Formula
        normalized_value = {(value − min) / (max - min)} * New_max

        Where:
            value = Original pixel intensity.
            min = Minimum intensity in the image. (Example: 50)
            max = Maximum intensity in the image. (Example: 200)
            new_max = Desired maximum value - (255 for 8-bit images).
            - (Because we will try to use maximum value to get maximum clarity)

        if A pixel with intensity (value) = 120
        Then
            normalized_value = {(120 − 50) / (200 - 50)} * 255 = 119

        Theory: 
        Why Normalize?
            1. Ensures pixel values are within a standard range (e.g., 0–255 for display).
            2. Helps in comparing different images fairly.
            3. Prevents overflow/underflow in computations.

        When is it Needed?
            1. After applying filters (e.g., Sobel edge detection, where gradients can be negative).
            2. When combining multiple images.
            3. Before saving an image to ensure correct pixel representation.
        """
        arr = arr.astype(np.float32)
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max == arr_min:
            return np.zeros_like(arr, dtype=np.uint8)
        return ((arr - arr_min) * (255 / (arr_max - arr_min))).clip(0, 255).astype(np.uint8)

    def safe_subtract(self, a, b):
        """Protected subtraction to prevent overflow"""
        return np.abs(a.astype(np.float32) - b.astype(np.float32))

    def simple_edge_detection(self, image, threshold=30):
        """
        What is "Extract 8-Neighborhood Pixels"?
        In image processing, the 8-neighborhood (or 8-connected neighborhood) of a pixel refers to the 8 surrounding pixels adjacent to it 
        (top-left, top, top-right, left, right, bottom-left, bottom, bottom-right).

        This forms a 3 X 3 grid centered at the target pixel:
        (i-1, j-1) | (i-1, j) | (i-1, j+1)  
        (i, j-1)   | (i, j)   | (i, j+1)  
        (i+1, j-1) | (i+1, j) | (i+1, j+1)


        Why Do We Need It?
        1. Detecting Local Intensity Changes
            Edges in an image occur where there is a sudden change in pixel intensity.
            By comparing a pixel with its 8 neighbors, we can determine if it lies on an edge.
            Example:
                If a pixel is much brighter/darker than most neighbors → likely an edge.
                If all neighbors have similar intensity → not an edge.

        2. Capturing Edges in All Directions
            A 4-neighborhood (top, bottom, left, right) would miss diagonal edges

        3. The maximum absolute difference between a pixel and its neighbors:
            If difference is more than thresold the draw a point other wise make it white.
            
            Edge(i,j) = { 255 if max(|I(i,j)-I(k,l)|) > Thresold  ∀(k,l) ∈ 8-neighborhood
            Edge(i,j) = { 0

            I(i,j) = Target pixel intensity.
            I(k,l) = Neighboring pixel intensity.
            T = Threshold (e.g., 30).
        """

        """Edge detection with overflow protection"""
        edges = np.zeros_like(image, dtype=np.uint8)
        rows, cols = image.shape
        
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j-1],                   image[i, j+1],
                    image[i+1, j-1], image[i+1, j], image[i+1, j+1]
                ]
                
                # Safe calculation of maximum difference
                max_diff = max(self.safe_subtract(image[i,j], neighbor) for neighbor in neighbors)
                
                if max_diff > threshold:
                    edges[i,j] = 255
                    
        return edges

    def sobel_edge_detection(self, image, threshold=100):
        """Edge Color Convention: Why White?
                Standard Practice:

                In most image processing systems, edges are white (255) on a black (0) background by convention.

                This mimics how edges would appear if you drew them with chalk (white) on a blackboard.

        Why Not Black Edges?

                Black edges (0) would blend into dark regions of the original image, making them harder to visualize.

                White edges provide maximum contrast against the black background.

        Mathematical Reason:

                Edge detection outputs a gradient magnitude (always ≥ 0).

                We set strong gradients (> threshold) to the maximum value (255 = white) and others to 0 (black).
                
        Simplified Sobel Edge Detection Explained
        What is the Sobel Operator?
                Imagine you're looking at a black-and-white photo. The Sobel operator helps you find the "edges" - places where the brightness changes suddenly, like where a dark object meets a light background.

                How It Works (3 Simple Steps):
                Two Magic Glasses (Sobel Kernels)

                We use two special 3×3 grids (called kernels):

                Horizontal (X) Kernel: Finds vertical edges

                text
                [-1, 0, +1]  
                [-2, 0, +2]  
                [-1, 0, +1]  
                Vertical (Y) Kernel: Finds horizontal edges

                text
                [-1, -2, -1]  
                [ 0,  0,  0]  
                [+1, +2, +1]  
                Slide and Multiply

                Take each 3×3 block of the image.

                Multiply pixel values with the kernel numbers, then add them up.

                Example for X-direction:

                text
                (Top-left pixel × -1) + (Top-middle × 0) + ... + (Bottom-right × +1)
                Combine Results

                Calculate edge strength:

                python
                edge_strength = sqrt(X_result² + Y_result²)
                If edge_strength > threshold, mark it as an edge (white), else background (black).     
                
        """

        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        gradient = np.zeros_like(image, dtype=np.float32)
        
        for i in range(1, image.shape[0]-1):
            for j in range(1, image.shape[1]-1):
                region = image[i-1:i+2, j-1:j+2].astype(np.float32)
                gx = np.sum(region * sobel_x)
                gy = np.sum(region * sobel_y)
                gradient[i,j] = np.sqrt(gx**2 + gy**2)
        
        return self.normalize(gradient) > threshold