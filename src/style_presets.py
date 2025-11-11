"""
Style presets management for paint-by-numbers generator.
Handles loading, auto-recommendation, and application of style presets.
"""

import yaml
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .config import ColorConfig


@dataclass
class StylePreset:
    """Style preset configuration."""
    name: str
    description: str
    recommended_for: List[str]
    palette: List[ColorConfig]
    processing: Dict[str, float]


class StyleManager:
    """Manages style presets and auto-recommendation."""
    
    def __init__(self, presets_file: str = "style_presets.yaml"):
        """Initialize style manager with presets file."""
        with open(presets_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        self.presets = {}
        for preset_name, preset_data in data['presets'].items():
            # Convert palette to ColorConfig objects
            palette = [ColorConfig(**color_data) for color_data in preset_data['palette']]
            
            self.presets[preset_name] = StylePreset(
                name=preset_data['name'],
                description=preset_data['description'],
                recommended_for=preset_data['recommended_for'],
                palette=palette,
                processing=preset_data['processing']
            )
        
        self.auto_rules = data['auto_recommendation']
    
    def get_preset(self, preset_name: str) -> Optional[StylePreset]:
        """Get a specific preset by name."""
        return self.presets.get(preset_name)
    
    def list_presets(self) -> List[str]:
        """List all available preset names."""
        return list(self.presets.keys())
    
    def recommend_style(self, image_array: np.ndarray) -> str:
        """
        Automatically recommend a style based on image analysis.
        
        Args:
            image_array: RGB image as numpy array
            
        Returns:
            Recommended preset name
        """
        height, width = image_array.shape[:2]
        aspect_ratio = height / width
        
        # Analyze image characteristics
        characteristics = self._analyze_image(image_array)
        
        # Apply recommendation rules
        if self._is_portrait(aspect_ratio, characteristics):
            return "portrait"
        elif self._is_landscape(aspect_ratio, characteristics):
            return "minimalist"  # Good for landscapes
        elif self._is_graphic(characteristics):
            return "minimalist"
        elif characteristics['colorfulness'] > 0.8:
            return "popart"
        elif characteristics['warmth'] > 0.6:
            return "vintage"
        else:
            return "portrait"  # Safe default
    
    def _analyze_image(self, image_array: np.ndarray) -> Dict[str, float]:
        """Analyze image characteristics for style recommendation."""
        # Convert to various color spaces for analysis
        hsv = self._rgb_to_hsv(image_array)
        gray = np.mean(image_array, axis=2)
        
        # Basic statistics
        characteristics = {
            'aspect_ratio': image_array.shape[0] / image_array.shape[1],
            'colorfulness': self._calculate_colorfulness(image_array),
            'brightness': np.mean(gray) / 255.0,
            'contrast': np.std(gray) / 255.0,
            'warmth': self._calculate_warmth(hsv),
            'edge_density': self._calculate_edge_density(gray),
            'green_blue_ratio': self._calculate_green_blue_ratio(hsv)
        }
        
        return characteristics
    
    def _rgb_to_hsv(self, rgb_array: np.ndarray) -> np.ndarray:
        """Convert RGB to HSV color space."""
        # Simple RGB to HSV conversion
        rgb = rgb_array.astype(np.float32) / 255.0
        max_val = np.max(rgb, axis=2)
        min_val = np.min(rgb, axis=2)
        delta = max_val - min_val
        
        # Hue
        hue = np.zeros_like(max_val)
        mask = delta != 0
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        
        hue[mask & (max_val == r)] = (g[mask & (max_val == r)] - b[mask & (max_val == r)]) / delta[mask & (max_val == r)] * 60
        hue[mask & (max_val == g)] = (b[mask & (max_val == g)] - r[mask & (max_val == g)]) / delta[mask & (max_val == g)] * 60 + 120
        hue[mask & (max_val == b)] = (r[mask & (max_val == b)] - g[mask & (max_val == b)]) / delta[mask & (max_val == b)] * 60 + 240
        hue = np.mod(hue, 360)
        
        # Saturation
        saturation = np.where(max_val == 0, 0, delta / max_val)
        
        # Value
        value = max_val
        
        return np.stack([hue, saturation * 255, value * 255], axis=2)
    
    def _calculate_colorfulness(self, image_array: np.ndarray) -> float:
        """Calculate how colorful an image is (0-1 scale)."""
        # Simple colorfulness metric based on RGB standard deviation
        r, g, b = image_array[:,:,0], image_array[:,:,1], image_array[:,:,2]
        
        # Calculate RG, YB differences
        rg = np.abs(r - g)
        yb = np.abs(0.5 * (r + g) - b)
        
        # Mean and standard deviation
        rg_mean, yb_mean = np.mean(rg), np.mean(yb)
        rg_std, yb_std = np.std(rg), np.std(yb)
        
        # Colorfulness metric
        colorfulness = np.sqrt(rg_std**2 + yb_std**2) + 0.3 * np.sqrt(rg_mean**2 + yb_mean**2)
        return min(colorfulness / 100, 1.0)  # Normalize to 0-1
    
    def _calculate_warmth(self, hsv_array: np.ndarray) -> float:
        """Calculate warmth ratio (warm colors vs cool colors)."""
        # Warm colors: red, orange, yellow (0-60 and 300-360 degrees)
        hue = hsv_array[:,:,0]
        warm_mask = ((hue >= 0) & (hue <= 60)) | ((hue >= 300) & (hue <= 360))
        return np.mean(warm_mask)
    
    def _calculate_edge_density(self, gray_array: np.ndarray) -> float:
        """Calculate edge density using simple gradient."""
        # Simple edge detection using Sobel
        from scipy import ndimage
        
        sx = ndimage.sobel(gray_array, axis=0, mode='constant')
        sy = ndimage.sobel(gray_array, axis=1, mode='constant')
        edges = np.hypot(sx, sy)
        return np.mean(edges > np.percentile(edges, 75))
    
    def _calculate_green_blue_ratio(self, hsv_array: np.ndarray) -> float:
        """Calculate ratio of green and blue pixels."""
        hue = hsv_array[:,:,0]
        green_mask = (hue >= 90) & (hue <= 150)  # Green range
        blue_mask = (hue >= 180) & (hue <= 260)  # Blue range
        
        green_ratio = np.mean(green_mask)
        blue_ratio = np.mean(blue_mask)
        return green_ratio + blue_ratio
    
    def _is_portrait(self, aspect_ratio: float, characteristics: Dict[str, float]) -> bool:
        """Check if image is likely portrait."""
        portrait_ratio = self.auto_rules['portrait_detection']['aspect_ratio_threshold']
        skin_tone_threshold = self.auto_rules['portrait_detection']['skin_tone_threshold']
        
        return (aspect_ratio > portrait_ratio and 
                characteristics['warmth'] > skin_tone_threshold)
    
    def _is_landscape(self, aspect_ratio: float, characteristics: Dict[str, float]) -> bool:
        """Check if image is likely landscape."""
        landscape_ratio = self.auto_rules['landscape_detection']['aspect_ratio_threshold']
        green_blue_threshold = self.auto_rules['landscape_detection']['green_blue_threshold']
        
        return (1/aspect_ratio > landscape_ratio and 
                characteristics['green_blue_ratio'] > green_blue_threshold)
    
    def _is_graphic(self, characteristics: Dict[str, float]) -> bool:
        """Check if image is likely a graphic/logo."""
        edge_threshold = self.auto_rules['graphic_detection']['edge_density_threshold']
        diversity_threshold = self.auto_rules['graphic_detection']['color_diversity_threshold']
        
        return (characteristics['edge_density'] > edge_threshold or 
                characteristics['colorfulness'] < diversity_threshold)
    
    def apply_processing_adjustments(self, image_array: np.ndarray, 
                                processing_settings: Dict[str, float]) -> np.ndarray:
        """
        Apply style-specific processing adjustments to image.
        
        Args:
            image_array: RGB image as numpy array
            processing_settings: Processing adjustments from preset
            
        Returns:
            Adjusted image array
        """
        adjusted = image_array.copy().astype(np.float32) / 255.0
        
        # Apply brightness adjustment
        brightness = processing_settings.get('brightness_adjustment', 0.0)
        if brightness != 0.0:
            adjusted = np.clip(adjusted + brightness, 0, 1)
        
        # Apply contrast enhancement
        contrast = processing_settings.get('contrast_enhancement', 1.0)
        if contrast != 1.0:
            mean_val = np.mean(adjusted)
            adjusted = np.clip((adjusted - mean_val) * contrast + mean_val, 0, 1)
        
        # Apply saturation boost
        saturation_boost = processing_settings.get('saturation_boost', 0.0)
        if saturation_boost != 0.0:
            hsv = self._rgb_to_hsv((adjusted * 255).astype(np.uint8))
            hsv[:,:,1] = np.clip(hsv[:,:,1] * (1 + saturation_boost), 0, 255)
            adjusted = self._hsv_to_rgb(hsv) / 255.0
        
        return (adjusted * 255).astype(np.uint8)
    
    def _hsv_to_rgb(self, hsv_array: np.ndarray) -> np.ndarray:
        """Convert HSV to RGB color space."""
        hue, saturation, value = hsv_array[:,:,0], hsv_array[:,:,1], hsv_array[:,:,2]
        
        # Normalize
        h = hue / 360.0
        s = saturation / 255.0
        v = value / 255.0
        
        # HSV to RGB conversion
        c = v * s
        x = c * (1 - np.abs((h / 60) % 2 - 1))
        m = v - c
        
        rgb = np.zeros_like(hsv_array)
        
        # Red channel
        mask = (h >= 0) & (h < 60)
        rgb[:,:,0][mask] = c[mask]
        mask = (h >= 60) & (h < 120)
        rgb[:,:,0][mask] = x[mask]
        mask = (h >= 120) & (h < 180)
        rgb[:,:,0][mask] = 0
        mask = (h >= 180) & (h < 240)
        rgb[:,:,0][mask] = -x[mask]
        mask = (h >= 240) & (h < 300)
        rgb[:,:,0][mask] = -c[mask]
        mask = (h >= 300) & (h < 360)
        rgb[:,:,0][mask] = x[mask]
        
        # Green channel
        mask = (h >= 0) & (h < 60)
        rgb[:,:,1][mask] = -x[mask]
        mask = (h >= 60) & (h < 120)
        rgb[:,:,1][mask] = c[mask]
        mask = (h >= 120) & (h < 180)
        rgb[:,:,1][mask] = x[mask]
        mask = (h >= 180) & (h < 240)
        rgb[:,:,1][mask] = 0
        mask = (h >= 240) & (h < 300)
        rgb[:,:,1][mask] = -c[mask]
        mask = (h >= 300) & (h < 360)
        rgb[:,:,1][mask] = x[mask]
        
        # Blue channel
        mask = (h >= 0) & (h < 60)
        rgb[:,:,2][mask] = -c[mask]
        mask = (h >= 60) & (h < 120)
        rgb[:,:,2][mask] = 0
        mask = (h >= 120) & (h < 180)
        rgb[:,:,2][mask] = c[mask]
        mask = (h >= 180) & (h < 240)
        rgb[:,:,2][mask] = x[mask]
        mask = (h >= 240) & (h < 300)
        rgb[:,:,2][mask] = -x[mask]
        mask = (h >= 300) & (h < 360)
        rgb[:,:,2][mask] = 0
        
        rgb = (rgb + m) * 255
        return np.clip(rgb, 0, 255).astype(np.uint8)
