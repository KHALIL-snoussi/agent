"""
DMC color palette management with CIEDE2000 color distance calculations.
"""

import csv
import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math


@dataclass
class DMCColor:
    """Represents a single DMC color with all necessary information."""
    dmc_code: str
    name: str
    rgb: Tuple[int, int, int]
    lab: Optional[Tuple[float, float, float]] = None
    hex: str = ""
    
    def __post_init__(self):
        """Calculate hex color and Lab coordinates."""
        self.hex = f"#{self.rgb[0]:02x}{self.rgb[1]:02x}{self.rgb[2]:02x}"
        if self.lab is None:
            self.lab = self._rgb_to_lab(self.rgb)
    
    @staticmethod
    def _rgb_to_lab(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert RGB to Lab color space."""
        # Normalize RGB to 0-1
        r, g, b = [c / 255.0 for c in rgb]
        
        # Apply gamma correction
        def gamma_correct(c):
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        
        r, g, b = gamma_correct(r), gamma_correct(g), gamma_correct(b)
        
        # Convert to XYZ (sRGB D65)
        x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
        
        # D65 white point
        x_ref, y_ref, z_ref = 95.047, 100.0, 108.883
        
        # Normalize
        x_ref, y_ref, z_ref = x_ref / 100.0, y_ref / 100.0, z_ref / 100.0
        x, y, z = x / 100.0, y / 100.0, z / 100.0
        
        # Lab conversion
        def lab_transform(t):
            return t ** (1/3) if t > 0.008856 else 7.787 * t + 16/116
        
        l = 116 * lab_transform(y / y_ref) - 16
        a = 500 * (lab_transform(x / x_ref) - lab_transform(y / y_ref))
        b_ = 200 * (lab_transform(y / y_ref) - lab_transform(z / z_ref))
        
        return (l, a, b_)


class CIEDE2000:
    """CIEDE2000 color difference calculation."""
    
    @staticmethod
    def delta_e2000(lab1: Tuple[float, float, float], 
                    lab2: Tuple[float, float, float]) -> float:
        """Calculate CIEDE2000 color difference between two Lab colors."""
        l1, a1, b1 = lab1
        l2, a2, b2 = lab2
        
        # Calculate CIELAB values
        c1 = math.sqrt(a1**2 + b1**2)
        c2 = math.sqrt(a2**2 + b2**2)
        c_mean = (c1 + c2) / 2
        
        # Calculate G factor
        g = 0.5 * (1 - math.sqrt(c_mean**7 / (c_mean**7 + 25**7)))
        
        # Adjust a values
        a1_prime = a1 * (1 + g)
        a2_prime = a2 * (1 + g)
        
        # Calculate C' values
        c1_prime = math.sqrt(a1_prime**2 + b1**2)
        c2_prime = math.sqrt(a2_prime**2 + b2**2)
        c_mean_prime = (c1_prime + c2_prime) / 2
        
        # Calculate h' values
        def h_prime(a, b):
            h = math.degrees(math.atan2(b, a))
            return h if h >= 0 else h + 360
        
        h1_prime = h_prime(a1_prime, b1)
        h2_prime = h_prime(a2_prime, b2)
        
        # Handle case where both colors are achromatic
        if c1_prime == 0 and c2_prime == 0:
            delta_h_prime = 0
            h_mean_prime = 0
        elif c1_prime * c2_prime == 0:
            delta_h_prime = 0
            h_mean_prime = h1_prime + h2_prime
        else:
            delta_h_prime = h2_prime - h1_prime
            if abs(delta_h_prime) <= 180:
                pass
            elif delta_h_prime > 180:
                delta_h_prime -= 360
            else:
                delta_h_prime += 360
            
            h_mean_prime = (h1_prime + h2_prime) / 2
            if abs(h1_prime - h2_prime) > 180:
                if h1_prime + h2_prime < 360:
                    h_mean_prime += 180
                else:
                    h_mean_prime -= 180
        
        # Calculate T factor
        t = (1 - 0.17 * math.cos(math.radians(h_mean_prime - 30)) +
              0.24 * math.cos(math.radians(2 * h_mean_prime)) +
              0.32 * math.cos(math.radians(3 * h_mean_prime + 6)) -
              0.20 * math.cos(math.radians(4 * h_mean_prime - 63)))
        
        # Calculate delta values
        delta_l_prime = l2 - l1
        delta_c_prime = c2_prime - c1_prime
        delta_h_prime = 2 * math.sqrt(c1_prime * c2_prime) * math.sin(math.radians(delta_h_prime / 2))
        
        # Calculate SL, SC, SH factors
        s_l = 1 + (0.015 * (l1 + l2) / 2)**2
        s_c = 1 + 0.045 * c_mean_prime
        s_h = 1 + 0.015 * c_mean_prime * t
        
        # Calculate RT factor
        delta_theta = 30 * math.exp(-((h_mean_prime - 275) / 25)**2)
        r_c = 2 * math.sqrt(c_mean_prime**7 / (c_mean_prime**7 + 25**7))
        r_t = -math.sin(math.radians(2 * delta_theta)) * r_c
        
        # Calculate final delta E
        k_l = k_c = k_h = 1  # Weighting factors (default)
        
        delta_e = math.sqrt(
            (delta_l_prime / (k_l * s_l))**2 +
            (delta_c_prime / (k_c * s_c))**2 +
            (delta_h_prime / (k_h * s_h))**2 +
            r_t * (delta_c_prime / (k_c * s_c)) * (delta_h_prime / (k_h * s_h))
        )
        
        return delta_e


class DMCPalette:
    """DMC color palette with efficient color matching."""
    
    def __init__(self, csv_path: str = "data/dmc.csv"):
        """Initialize palette from CSV file."""
        self.colors: List[DMCColor] = []
        self.dmc_lookup: Dict[str, DMCColor] = {}
        self.lab_array: Optional[np.ndarray] = None
        
        if os.path.exists(csv_path):
            self.load_from_csv(csv_path)
        else:
            raise FileNotFoundError(f"DMC palette file not found: {csv_path}")
    
    def load_from_csv(self, csv_path: str):
        """Load DMC colors from CSV file."""
        self.colors.clear()
        self.dmc_lookup.clear()
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        # Skip empty rows
                        if not row or row.get('dmc') is None:
                            continue
                        
                        dmc_code = row['dmc'].strip()
                        name = row.get('name', '').strip() if row.get('name') else ''
                        
                        # Skip if RGB values are missing
                        if row.get('r') is None or row.get('g') is None or row.get('b') is None:
                            continue
                        
                        r = int(row['r'])
                        g = int(row['g'])
                        b = int(row['b'])
                        
                        color = DMCColor(dmc_code, name, (r, g, b))
                        self.colors.append(color)
                        self.dmc_lookup[dmc_code] = color
                        
                    except (ValueError, KeyError) as e:
                        print(f"Warning: Skipping invalid DMC entry: {row} - {e}")
        except UnicodeDecodeError:
            # Fallback to latin-1 encoding if utf-8 fails
            with open(csv_path, 'r', encoding='latin-1') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        # Skip empty rows
                        if not row or row.get('dmc') is None:
                            continue
                        
                        dmc_code = row['dmc'].strip()
                        name = row.get('name', '').strip() if row.get('name') else ''
                        
                        # Skip if RGB values are missing
                        if row.get('r') is None or row.get('g') is None or row.get('b') is None:
                            continue
                        
                        r = int(row['r'])
                        g = int(row['g'])
                        b = int(row['b'])
                        
                        color = DMCColor(dmc_code, name, (r, g, b))
                        self.colors.append(color)
                        self.dmc_lookup[dmc_code] = color
                        
                    except (ValueError, KeyError) as e:
                        print(f"Warning: Skipping invalid DMC entry: {row} - {e}")
        
        # Pre-compute Lab array for fast nearest neighbor search
        self._compute_lab_array()
        
        print(f"Loaded {len(self.colors)} DMC colors from {csv_path}")
    
    def _compute_lab_array(self):
        """Pre-compute Lab coordinates for efficient searching."""
        if self.colors:
            self.lab_array = np.array([color.lab for color in self.colors])
    
    def find_nearest_color(self, rgb: Tuple[int, int, int], 
                         preserve_skin_tones: bool = False,
                         is_skin_region: bool = False) -> DMCColor:
        """Find nearest DMC color to given RGB using CIEDE2000."""
        target_lab = DMCColor._rgb_to_lab(rgb)
        
        if self.lab_array is None:
            raise RuntimeError("Lab array not initialized")
        
        # Calculate CIEDE2000 distances to all colors
        min_distance = float('inf')
        best_color = self.colors[0]
        
        for i, color in enumerate(self.colors):
            distance = CIEDE2000.delta_e2000(target_lab, color.lab)
            
            # Apply skin tone preservation if requested
            if preserve_skin_tones and is_skin_region:
                # Penalize non-skin colors for skin regions
                if not self._is_skin_tone_color(color):
                    distance *= 1.5  # Penalty factor
            
            if distance < min_distance:
                min_distance = distance
                best_color = color
        
        return best_color
    
    def find_nearest_color_fast(self, rgb: Tuple[int, int, int]) -> DMCColor:
        """Fast nearest color search using pre-computed Lab array."""
        target_lab = DMCColor._rgb_to_lab(rgb)
        target_lab_array = np.array([target_lab])
        
        if self.lab_array is None:
            raise RuntimeError("Lab array not initialized")
        
        # Calculate Euclidean distances in Lab space (faster approximation)
        distances = np.linalg.norm(self.lab_array - target_lab_array, axis=1)
        best_idx = np.argmin(distances)
        
        return self.colors[best_idx]
    
    def _is_skin_tone_color(self, color: DMCColor) -> bool:
        """Check if a DMC color is likely a skin tone."""
        r, g, b = color.rgb
        
        # Simple skin tone detection based on RGB ratios
        # Skin tones typically have: R > G > B, and specific ranges
        if not (r > g > b):
            return False
        
        # Check if color falls within typical skin tone ranges
        rg_ratio = r / g if g > 0 else 0
        gb_ratio = g / b if b > 0 else 0
        
        # Typical skin tone ratios
        skin_r_range = (150, 255)
        skin_g_range = (100, 200)
        skin_b_range = (80, 180)
        rg_skin_range = (1.0, 1.3)
        gb_skin_range = (1.0, 1.5)
        
        return (skin_r_range[0] <= r <= skin_r_range[1] and
                skin_g_range[0] <= g <= skin_g_range[1] and
                skin_b_range[0] <= b <= skin_b_range[1] and
                rg_skin_range[0] <= rg_ratio <= rg_skin_range[1] and
                gb_skin_range[0] <= gb_ratio <= gb_skin_range[1])
    
    def get_color_by_code(self, dmc_code: str) -> Optional[DMCColor]:
        """Get DMC color by code."""
        return self.dmc_lookup.get(dmc_code)
    
    def get_color_count(self) -> int:
        """Get total number of colors in palette."""
        return len(self.colors)
    
    def get_skin_tone_colors(self) -> List[DMCColor]:
        """Get all colors that could be skin tones."""
        return [color for color in self.colors if self._is_skin_tone_color(color)]
    
    def export_to_dict(self) -> dict:
        """Export palette to dictionary format."""
        return {
            color.dmc_code: {
                'name': color.name,
                'rgb': color.rgb,
                'hex': color.hex,
                'lab': color.lab
            }
            for color in self.colors
        }


# Global palette instance
_dmc_palette: Optional[DMCPalette] = None


def get_dmc_palette(csv_path: str = None) -> DMCPalette:
    """Get global DMC palette instance."""
    global _dmc_palette
    if _dmc_palette is None:
        if csv_path is None:
            # Find DMC CSV file relative to project root
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up from src/diamondkit to project root
            project_root = os.path.dirname(os.path.dirname(current_dir))
            csv_path = os.path.join(project_root, "data", "dmc.csv")
        _dmc_palette = DMCPalette(csv_path)
    return _dmc_palette


def nearest_dmc(rgb: Tuple[int, int, int], 
               preserve_skin_tones: bool = False,
               is_skin_region: bool = False) -> DMCColor:
    """Find nearest DMC color to RGB."""
    palette = get_dmc_palette()
    return palette.find_nearest_color(rgb, preserve_skin_tones, is_skin_region)
