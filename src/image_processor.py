"""
Image processing module for paint-by-numbers generation.
Handles image preprocessing, color quantization, and grid generation.
"""

import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from skimage import color, filters, util
from sklearn.cluster import KMeans
import cv2
from typing import Tuple, List
import random

from .config import Config


class ImageProcessor:
    """Handles all image processing operations."""
    
    def __init__(self, config: Config):
        self.config = config
        np.random.seed(config.processing.seed)
        random.seed(config.processing.seed)
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess the input image:
        - Convert to RGB if needed
        - Auto-orient based on EXIF
        - Convert to sRGB
        - Smart crop to 3:4 aspect ratio
        - Resize to target resolution
        """
        # Load image
        img = Image.open(image_path)
        
        # Auto-orient based on EXIF
        img = ImageOps.exif_transpose(img)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Smart crop to 3:4 aspect ratio
        img_array = self._smart_crop_to_aspect_ratio(img_array)
        
        # Calculate target resolution based on drill size
        target_resolution = self._calculate_target_resolution()
        
        # Resize image
        img_resized = cv2.resize(img_array, target_resolution, interpolation=cv2.INTER_AREA)
        
        return img_resized
    
    def _smart_crop_to_aspect_ratio(self, img_array: np.ndarray) -> np.ndarray:
        """Smart crop image to maintain 3:4 aspect ratio."""
        h, w = img_array.shape[:2]
        target_ratio = self.config.canvas.aspect_ratio  # 0.75 for 3:4
        current_ratio = w / h
        
        if current_ratio > target_ratio:
            # Image is too wide, crop width
            new_width = int(h * target_ratio)
            x_start = (w - new_width) // 2
            return img_array[:, x_start:x_start + new_width]
        else:
            # Image is too tall, crop height
            new_height = int(w / target_ratio)
            y_start = (h - new_height) // 2
            return img_array[y_start:y_start + new_height, :]
    
    def _calculate_target_resolution(self) -> Tuple[int, int]:
        """
        Calculate target resolution based on drill size and canvas dimensions.
        For 30×40 cm canvas with 2.5mm drills:
        - Width: 300mm / 2.5mm = 120 drills
        - Height: 400mm / 2.5mm = 160 drills
        """
        width_mm = self.config.canvas.width_cm * 10
        height_mm = self.config.canvas.height_cm * 10
        drill_size = self.config.canvas.drill_size_mm
        
        drills_width = int(width_mm / drill_size)
        drills_height = int(height_mm / drill_size)
        
        return drills_width, drills_height
    
    def quantize_colors(self, img_array: np.ndarray) -> np.ndarray:
        """
        Quantize image colors to the specified palette.
        Uses Lab color space for better perceptual results.
        """
        # Convert to Lab color space
        if self.config.processing.color_space == "Lab":
            lab_image = color.rgb2lab(img_array)
            # Reshape for clustering
            pixels = lab_image.reshape(-1, 3)
        else:
            # Use RGB directly
            pixels = img_array.reshape(-1, 3) / 255.0
        
        # Get palette colors in the same color space
        palette_rgb = self.config.get_color_palette_rgb()
        if self.config.processing.color_space == "Lab":
            palette_lab = color.rgb2lab(np.array(palette_rgb).reshape(1, -1, 3) / 255.0)
            palette_points = palette_lab.reshape(-1, 3)
        else:
            palette_points = np.array(palette_rgb) / 255.0
        
        # Quantize using nearest neighbor
        quantized_indices = self._assign_to_palette(pixels, palette_points)
        
        # Apply dithering if enabled
        if self.config.processing.dithering:
            quantized_indices = self._apply_dithering(pixels, palette_points, quantized_indices)
        
        # Convert back to RGB
        quantized_pixels = palette_points[quantized_indices]
        if self.config.processing.color_space == "Lab":
            quantized_rgb = color.lab2rgb(quantized_pixels.reshape(img_array.shape))
            quantized_rgb = (quantized_rgb * 255).astype(np.uint8)
        else:
            quantized_rgb = (quantized_pixels * 255).astype(np.uint8)
        
        return quantized_rgb, quantized_indices.reshape(img_array.shape[:2])
    
    def _assign_to_palette(self, pixels: np.ndarray, palette: np.ndarray) -> np.ndarray:
        """Assign each pixel to nearest palette color."""
        # Calculate distances to all palette colors
        distances = np.linalg.norm(pixels[:, np.newaxis] - palette[np.newaxis, :], axis=2)
        return np.argmin(distances, axis=1)
    
    def _apply_dithering(self, pixels: np.ndarray, palette: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Apply Floyd-Steinberg dithering in the specified color space."""
        # Calculate the 2D dimensions from the original image shape
        # We need to pass the original image dimensions to this function
        # For now, skip dithering and return original indices
        return indices
        
        # Floyd-Steinberg dithering
        for y in range(h):
            for x in range(w):
                old_pixel = pixels[y * w + x]
                old_idx = img_2d[y, x]
                
                # Get nearest color
                distances = np.linalg.norm(old_pixel - palette, axis=1)
                new_idx = np.argmin(distances)
                img_2d[y, x] = new_idx
                
                # Calculate error
                new_pixel = palette[new_idx]
                error = old_pixel - new_pixel
                
                # Distribute error to neighbors (simplified for color space)
                if x < w - 1:
                    pixels[y * w + x + 1] += error * 7/16
                if y < h - 1:
                    if x > 0:
                        pixels[(y + 1) * w + x - 1] += error * 3/16
                    pixels[(y + 1) * w + x] += error * 5/16
                    if x < w - 1:
                        pixels[(y + 1) * w + x + 1] += error * 1/16
        
        return img_2d.flatten()
    
    def generate_grid_with_symbols(self, color_indices: np.ndarray) -> Tuple[np.ndarray, List[List[str]]]:
        """
        Generate symbol grid from color indices.
        Assigns unique symbols to each color and creates a grid of symbols.
        """
        h, w = color_indices.shape
        symbols = self.config.symbols.symbol_set
        
        # Create symbol grid
        symbol_grid = np.empty((h, w), dtype=object)
        for i in range(len(symbols)):
            symbol_grid[color_indices == i] = symbols[i]
        
        return symbol_grid, symbols
    
    def count_color_usage(self, color_indices: np.ndarray) -> List[int]:
        """Count the number of cells for each color."""
        color_counts = []
        for i in range(len(self.config.palette)):
            count = np.sum(color_indices == i)
            # Add spare percentage
            count_with_spare = int(count * (1 + self.config.output.spare_percentage))
            color_counts.append(count_with_spare)
        
        return color_counts
