"""
Quality assessment module for paint-by-numbers generator.
Evaluates image quality and provides feedback for improvement.
"""

import numpy as np
from typing import Dict, Tuple, List
from enum import Enum
import cv2

class QualityLevel(Enum):
    """Quality assessment levels."""
    GOOD = "Good"
    OK = "OK"
    LOW = "Low"

class QualityAssessment:
    """Quality assessment result."""
    def __init__(self, level: QualityLevel, score: float, issues: List[str], suggestions: List[str]):
        self.level = level
        self.score = score
        self.issues = issues
        self.suggestions = suggestions

class QualityAssessor:
    """Assesses image quality for paint-by-numbers conversion."""
    
    def __init__(self):
        self.min_resolution = (120, 160)  # Target resolution for 30x40cm canvas
        self.min_contrast = 0.3
        self.color_diversity_threshold = 0.4
    
    def assess_quality(self, original_image: np.ndarray, 
                    quantized_image: np.ndarray,
                    palette_size: int = 7) -> QualityAssessment:
        """
        Assess image quality and provide feedback.
        
        Args:
            original_image: Original RGB image array
            quantized_image: Quantized image array
            palette_size: Number of colors in palette
            
        Returns:
            QualityAssessment object
        """
        issues = []
        suggestions = []
        
        # Analyze various quality metrics
        resolution_score = self._assess_resolution(original_image.shape[:2])
        contrast_score = self._assess_contrast(original_image)
        diversity_score = self._assess_color_diversity(original_image)
        quantization_score = self._assess_quantization_quality(original_image, quantized_image)
        
        # Calculate overall score
        weights = {
            'resolution': 0.25,
            'contrast': 0.25,
            'diversity': 0.25,
            'quantization': 0.25
        }
        
        overall_score = (
            resolution_score * weights['resolution'] +
            contrast_score * weights['contrast'] +
            diversity_score * weights['diversity'] +
            quantization_score * weights['quantization']
        )
        
        # Determine quality level
        if overall_score >= 0.8:
            level = QualityLevel.GOOD
        elif overall_score >= 0.6:
            level = QualityLevel.OK
        else:
            level = QualityLevel.LOW
        
        # Generate issues and suggestions
        if resolution_score < 0.7:
            issues.append("Low resolution")
            suggestions.append("Use higher resolution image (minimum 1200x1600 pixels)")
        
        if contrast_score < 0.6:
            issues.append("Low contrast")
            suggestions.append("Increase image contrast before processing")
        
        if diversity_score < 0.5:
            issues.append("Limited color variety")
            suggestions.append("Try a more colorful image or Pop Art style")
        
        if quantization_score < 0.6:
            issues.append("Poor quantization result")
            suggestions.append("Try different style preset or crop to main subject")
        
        # Specific suggestions based on image characteristics
        self._add_specific_suggestions(original_image, issues, suggestions)
        
        return QualityAssessment(level, overall_score, issues, suggestions)
    
    def _assess_resolution(self, shape: Tuple[int, int]) -> float:
        """Assess image resolution adequacy."""
        height, width = shape
        min_height, min_width = self.min_resolution
        
        if width >= min_width and height >= min_height:
            # Good resolution
            ratio = min(width / min_width, height / min_height)
            return min(ratio, 1.0)
        else:
            # Low resolution
            width_score = min(width / min_width, 1.0)
            height_score = min(height / min_height, 1.0)
            return (width_score + height_score) / 2
    
    def _assess_contrast(self, image: np.ndarray) -> float:
        """Assess image contrast using RMS contrast."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate RMS contrast
        mean_intensity = np.mean(gray)
        rms_contrast = np.sqrt(np.mean((gray - mean_intensity) ** 2))
        
        # Normalize to 0-1 scale (higher is better contrast)
        normalized_contrast = min(rms_contrast / 64.0, 1.0)
        return normalized_contrast
    
    def _assess_color_diversity(self, image: np.ndarray) -> float:
        """Assess color diversity in the image."""
        # Reshape image to list of pixels
        pixels = image.reshape(-1, 3)
        
        # Calculate unique colors
        unique_colors = len(np.unique(pixels, axis=0))
        total_pixels = len(pixels)
        
        # Color diversity ratio
        diversity_ratio = unique_colors / total_pixels
        
        # Normalize against ideal diversity
        ideal_diversity = 0.1  # 10% of pixels should be unique
        return min(diversity_ratio / ideal_diversity, 1.0)
    
    def _assess_quantization_quality(self, original: np.ndarray, 
                               quantized: np.ndarray) -> float:
        """Assess how well quantization preserved the original."""
        # Resize images to match if they don't
        if original.shape != quantized.shape:
            from PIL import Image
            orig_img = Image.fromarray(original)
            quant_img = Image.fromarray(quantized)
            
            # Resize quantized to original size for comparison
            quant_img = quant_img.resize(orig_img.size, Image.Resampling.LANCZOS)
            quantized = np.array(quant_img)
        
        # Calculate color difference between original and quantized
        original_lab = self._rgb_to_lab(original)
        quantized_lab = self._rgb_to_lab(quantized)
        
        # Calculate Delta E (color difference)
        delta_e = np.mean(np.linalg.norm(original_lab - quantized_lab, axis=2))
        
        # Normalize (lower Delta E is better)
        max_acceptable_delta_e = 15.0
        quality = max(0, 1.0 - delta_e / max_acceptable_delta_e)
        return quality
    
    def _rgb_to_lab(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to Lab color space."""
        # Simple approximation for quality assessment
        rgb_norm = rgb.astype(np.float32) / 255.0
        
        # Convert to XYZ (simplified)
        r, g, b = rgb_norm[:,:,0], rgb_norm[:,:,1], rgb_norm[:,:,2]
        
        # Apply gamma correction
        r = np.where(r > 0.04045, ((r + 0.055) / 1.055) ** 2.4, r / 12.92)
        g = np.where(g > 0.04045, ((g + 0.055) / 1.055) ** 2.4, g / 12.92)
        b = np.where(b > 0.04045, ((b + 0.055) / 1.055) ** 2.4, b / 12.92)
        
        # XYZ conversion matrices
        x = r * 0.4124 + g * 0.3576 + b * 0.1805
        y = r * 0.2126 + g * 0.7152 + b * 0.0722
        z = r * 0.0193 + g * 0.1192 + b * 0.9505
        
        # Lab conversion
        x_ref, y_ref, z_ref = 95.047, 100.0, 108.883
        
        # L* component
        l = np.where(y > 0.008856, 116 * (y / y_ref) ** (1/3) - 16, 903.3 * y / y_ref)
        
        # a* component
        a = 500 * (self._f(x / x_ref) - self._f(y / y_ref))
        
        # b* component
        b = 200 * (self._f(y / y_ref) - self._f(z / z_ref))
        
        return np.stack([l, a, b], axis=2)
    
    def _f(self, t: np.ndarray) -> np.ndarray:
        """Helper function for Lab conversion."""
        delta = 6.0 / 29.0
        return np.where(t > delta ** 3, t ** (1/3), t / (3 * delta ** 2) + 4/29)
    
    def _add_specific_suggestions(self, image: np.ndarray, 
                            issues: List[str], suggestions: List[str]):
        """Add specific suggestions based on image analysis."""
        height, width = image.shape[:2]
        aspect_ratio = height / width
        
        # Aspect ratio suggestions
        if aspect_ratio > 1.5:  # Very tall portrait
            suggestions.append("Consider cropping to focus on main subject")
        
        # Brightness suggestions
        brightness = np.mean(image)
        if brightness < 80:  # Dark image
            suggestions.append("Brighten image before processing for better results")
        elif brightness > 200:  # Overexposed
            suggestions.append("Reduce brightness to reveal more details")
        
        # Color dominance suggestions
        color_means = np.mean(image, axis=(0, 1))
        dominant_color = np.argmax(color_means)
        
        if dominant_color == 0:  # Red dominant
            suggestions.append("Portrait style recommended for red-heavy images")
        elif dominant_color == 1:  # Green dominant
            suggestions.append("Minimalist style works well with green landscapes")
        elif dominant_color == 2:  # Blue dominant
            suggestions.append("Vintage style complements blue-heavy images")
    
    def generate_quality_report(self, assessment: QualityAssessment) -> str:
        """Generate a formatted quality report."""
        report = f"""
📊 Quality Assessment Report
{"="*40}

Overall Quality: {assessment.level.value}
Score: {assessment.score:.2f}/1.00

Issues Found:
"""
        if assessment.issues:
            for issue in assessment.issues:
                report += f"  • {issue}\n"
        else:
            report += "  ✅ No major issues detected\n"
        
        report += "\nSuggestions for Improvement:\n"
        for suggestion in assessment.suggestions:
            report += f"  💡 {suggestion}\n"
        
        report += f"""
{"="*40}

Quality Guide:
• Good (0.8-1.0): Optimal for paint-by-numbers
• OK (0.6-0.8): Acceptable with minor compromises  
• Low (0.0-0.6): Consider improvements before processing
"""
        return report
