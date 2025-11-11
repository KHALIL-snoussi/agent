"""
Image loading and preprocessing for diamond painting kit generation.
"""

import os
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from typing import Tuple, Optional
import cv2

from .config import Config


class ImageLoader:
    """Handles image loading, preprocessing, and color space conversions."""
    
    def __init__(self, config: Config):
        """Initialize image loader with configuration."""
        self.config = config
    
    def load_image(self, image_path: str) -> Tuple[np.ndarray, dict]:
        """
        Load and preprocess input image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (processed_image_lab, metadata)
        """
        print(f"Loading image: {image_path}")
        
        # Load image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load with PIL
        pil_image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Store original metadata
        metadata = {
            'original_size': pil_image.size,
            'original_mode': pil_image.mode,
            'filename': os.path.basename(image_path)
        }
        
        # Auto-orient image based on EXIF
        pil_image = ImageOps.exif_transpose(pil_image)
        
        # Convert to numpy array
        rgb_image = np.array(pil_image)
        
        # Preprocess image
        processed_image = self._preprocess_image(rgb_image)
        
        # Convert to target canvas aspect ratio
        canvas_aspect = self.config.canvas.aspect_ratio
        processed_image = self._resize_to_aspect_ratio(processed_image, canvas_aspect)
        
        # Convert to Lab color space
        lab_image = self._rgb_to_lab(processed_image)
        
        # Update metadata
        metadata.update({
            'processed_size': (processed_image.shape[1], processed_image.shape[0]),
            'final_size': (lab_image.shape[1], lab_image.shape[0]),
            'color_space': 'Lab'
        })
        
        print(f"Image processed: {metadata['final_size'][0]}×{metadata['final_size'][1]} pixels")
        
        return lab_image, metadata
    
    def _preprocess_image(self, rgb_image: np.ndarray) -> np.ndarray:
        """Apply image preprocessing enhancements."""
        processed = rgb_image.copy()
        
        # Apply gentle contrast enhancement
        enhancer = ImageEnhance.Contrast(Image.fromarray(processed))
        processed = np.array(enhancer.enhance(1.1))
        
        # Apply subtle sharpening
        enhancer = ImageEnhance.Sharpness(Image.fromarray(processed))
        processed = np.array(enhancer.enhance(1.05))
        
        # Normalize exposure if needed
        processed = self._normalize_exposure(processed)
        
        return processed
    
    def _normalize_exposure(self, image: np.ndarray) -> np.ndarray:
        """Normalize image exposure using histogram stretching."""
        # Convert to float for processing
        img_float = image.astype(np.float32) / 255.0
        
        # Apply simple histogram normalization to avoid CLAHE issues
        # This is safer and works with all image types
        normalized = img_float.copy()
        
        # Simple contrast stretching
        for channel in range(3):
            channel_data = img_float[:, :, channel]
            channel_min, channel_max = np.min(channel_data), np.max(channel_data)
            if channel_max > channel_min:
                normalized[:, :, channel] = (channel_data - channel_min) / (channel_max - channel_min)
        
        # Convert back to uint8
        return (normalized * 255).astype(np.uint8)
    
    def _resize_to_aspect_ratio(self, image: np.ndarray, target_aspect: float) -> np.ndarray:
        """Resize image to match target aspect ratio using smart cropping."""
        h, w = image.shape[:2]
        current_aspect = w / h
        
        if abs(current_aspect - target_aspect) < 0.01:  # Within 1% tolerance
            return image
        
        print(f"Adjusting aspect ratio from {current_aspect:.3f} to {target_aspect:.3f}")
        
        if current_aspect > target_aspect:
            # Image is too wide - crop width
            new_width = int(h * target_aspect)
            x_start = (w - new_width) // 2
            x_end = x_start + new_width
            cropped = image[:, x_start:x_end]
        else:
            # Image is too tall - crop height
            new_height = int(w / target_aspect)
            y_start = (h - new_height) // 2
            y_end = y_start + new_height
            cropped = image[y_start:y_end, :]
        
        print(f"Cropped to {cropped.shape[1]}×{cropped.shape[0]} pixels")
        return cropped
    
    def _rgb_to_lab(self, rgb_image: np.ndarray) -> np.ndarray:
        """Convert RGB image to Lab color space."""
        # Normalize to 0-1 range with explicit float32
        rgb_float = rgb_image.astype(np.float32) / 255.0
        
        # Convert to Lab using OpenCV, ensure output is float32
        lab_image = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2LAB)
        lab_image = lab_image.astype(np.float32)  # Ensure float32 consistency
        
        # Scale Lab to standard ranges
        lab_image[:, :, 0] = lab_image[:, :, 0] * 100.0 / 255.0  # L: 0-100
        # a and b channels are already in approximate range
        
        return lab_image
    
    def lab_to_rgb(self, lab_image: np.ndarray) -> np.ndarray:
        """Convert Lab image back to RGB color space."""
        # Scale Lab back to OpenCV ranges, ensure float32
        lab_copy = lab_image.copy().astype(np.float32)
        lab_copy[:, :, 0] = lab_copy[:, :, 0] * 255.0 / 100.0  # L: back to 0-255
        # a and b channels stay as-is
        
        # Convert to RGB using OpenCV, ensure float32 output
        rgb_float = cv2.cvtColor(lab_copy, cv2.COLOR_LAB2RGB)
        rgb_float = rgb_float.astype(np.float32)  # Ensure float32 consistency
        
        # Convert back to 0-255 uint8
        return (rgb_float * 255).astype(np.uint8)
    
    def validate_image(self, image_path: str) -> dict:
        """
        Validate input image for processing.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                result['errors'].append(f"File not found: {image_path}")
                return result
            
            # Check file size
            file_size = os.path.getsize(image_path)
            if file_size > 50 * 1024 * 1024:  # 50MB limit
                result['errors'].append(f"File too large: {file_size / (1024*1024):.1f}MB (max 50MB)")
                return result
            
            # Try to load image
            with Image.open(image_path) as img:
                result['info']['size'] = img.size
                result['info']['mode'] = img.mode
                result['info']['format'] = img.format
                
                # Check dimensions
                w, h = img.size
                if w < 100 or h < 100:
                    result['errors'].append(f"Image too small: {w}×{h} (minimum 100×100)")
                elif w > 8000 or h > 8000:
                    result['warnings'].append(f"Image very large: {w}×{h} (processing may be slow)")
                
                # Check mode
                if img.mode not in ['RGB', 'RGBA', 'L']:
                    result['warnings'].append(f"Unusual image mode: {img.mode}")
                
                # Check aspect ratio compatibility
                current_aspect = w / h
                target_aspect = self.config.canvas.aspect_ratio
                aspect_diff = abs(current_aspect - target_aspect) / target_aspect
                
                if aspect_diff > 0.5:  # More than 50% difference
                    result['warnings'].append(
                        f"Significant aspect ratio difference: {current_aspect:.2f} vs {target_aspect:.2f}"
                    )
                elif aspect_diff > 0.2:  # More than 20% difference
                    result['warnings'].append(
                        f"Moderate aspect ratio difference: {current_aspect:.2f} vs {target_aspect:.2f}"
                    )
            
            result['valid'] = True
            
        except Exception as e:
            result['errors'].append(f"Failed to load image: {str(e)}")
        
        return result
    
    def get_image_info(self, image_path: str) -> dict:
        """Get detailed information about an image file."""
        if not os.path.exists(image_path):
            return {'error': 'File not found'}
        
        try:
            with Image.open(image_path) as img:
                # Convert to RGB for consistent analysis
                rgb_img = img.convert('RGB') if img.mode != 'RGB' else img
                
                # Basic info
                info = {
                    'filename': os.path.basename(image_path),
                    'size': rgb_img.size,
                    'mode': img.mode,
                    'format': img.format,
                    'file_size': os.path.getsize(image_path)
                }
                
                # Calculate statistics
                img_array = np.array(rgb_img)
                info['mean_color'] = tuple(np.mean(img_array, axis=(0, 1)).astype(int))
                info['std_color'] = tuple(np.std(img_array, axis=(0, 1)).astype(int))
                
                # Aspect ratio
                w, h = rgb_img.size
                info['aspect_ratio'] = w / h
                
                # Color diversity (rough estimate)
                unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
                info['unique_colors'] = unique_colors
                
                # Estimate difficulty
                if unique_colors < 1000:
                    difficulty = "Easy"
                elif unique_colors < 5000:
                    difficulty = "Medium"
                else:
                    difficulty = "Hard"
                
                info['estimated_difficulty'] = difficulty
                
                return info
                
        except Exception as e:
            return {'error': str(e)}
    
    def save_processed_image(self, lab_image: np.ndarray, output_path: str):
        """Save processed Lab image as RGB for debugging."""
        rgb_image = self.lab_to_rgb(lab_image)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save image
        Image.fromarray(rgb_image).save(output_path)
        print(f"Saved processed image: {output_path}")
