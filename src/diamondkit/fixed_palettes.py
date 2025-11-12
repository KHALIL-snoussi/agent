"""
Fixed 7-color DMC palettes for commercial diamond painting kits.
Each style uses its own static palette that never changes.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from .dmc import DMCColor, get_dmc_palette


@dataclass
class FixedPalette:
    """Represents a fixed 7-color DMC palette."""
    name: str
    description: str
    rationale: str
    dmc_codes: List[str]
    
    def __post_init__(self):
        """Validate that exactly 7 DMC codes are provided."""
        if len(self.dmc_codes) != 7:
            raise ValueError(f"Palette {self.name} must have exactly 7 DMC codes, got {len(self.dmc_codes)}")
    
    def get_colors(self) -> List[DMCColor]:
        """Get DMCColor objects for this palette."""
        dmc_palette = get_dmc_palette()
        colors = []
        
        for code in self.dmc_codes:
            color = dmc_palette.get_color_by_code(code)
            if color is None:
                raise ValueError(f"DMC code {code} not found in database for palette {self.name}")
            colors.append(color)
        
        return colors


class FixedPaletteManager:
    """Manages the three fixed palettes as specified in requirements."""
    
    def __init__(self):
        """Initialize with the three required palettes."""
        self.palettes = self._create_palettes()
        self._validate_all_codes()
    
    def _create_palettes(self) -> Dict[str, FixedPalette]:
        """Create the three required fixed palettes."""
        return {
            "ORIGINAL": FixedPalette(
                name="ORIGINAL",
                description="Balanced palette with primary colors and neutrals",
                rationale="black/white anchors; primary red/yellow/blue; vivid green for nature; warm tan for skin/earth midtones",
                dmc_codes=["310", "B5200", "321", "444", "700", "797", "738"]
            ),
            
            "VINTAGE": FixedPalette(
                name="VINTAGE", 
                description="Muted sepia/cream range for traditional looks",
                rationale="muted sepia/cream range; deep brown anchor (3371); soft cream (3865); warm browns/beiges + mustard",
                dmc_codes=["3371", "3865", "801", "613", "3033", "372", "3790"]
            ),
            
            "POPART": FixedPalette(
                name="POPART",
                description="Bold high-contrast colors for vibrant designs", 
                rationale="bold high-contrast set; black/white; bright red/yellow/green; electric blue (996); hot magenta (915)",
                dmc_codes=["310", "B5200", "666", "444", "700", "996", "915"]
            )
        }
    
    def _validate_all_codes(self):
        """Validate that all DMC codes exist in the database."""
        dmc_palette = get_dmc_palette()
        missing_codes = []
        
        for palette_name, palette in self.palettes.items():
            for code in palette.dmc_codes:
                if dmc_palette.get_color_by_code(code) is None:
                    missing_codes.append(f"{palette_name}:{code}")
        
        if missing_codes:
            raise ValueError(f"Missing DMC codes in database: {missing_codes}")
    
    def get_palette(self, style_name: str) -> FixedPalette:
        """Get a specific palette by style name."""
        if style_name not in self.palettes:
            raise ValueError(f"Unknown style '{style_name}'. Available: {list(self.palettes.keys())}")
        return self.palettes[style_name]
    
    def get_palette_colors(self, style_name: str) -> List[DMCColor]:
        """Get DMCColor list for a specific style."""
        palette = self.get_palette(style_name)
        return palette.get_colors()
    
    def list_styles(self) -> List[str]:
        """Get list of available style names."""
        return list(self.palettes.keys())
    
    def get_style_info(self) -> Dict[str, Dict]:
        """Get detailed information about all styles."""
        info = {}
        for name, palette in self.palettes.items():
            colors = palette.get_colors()
            info[name] = {
                "name": palette.name,
                "description": palette.description,
                "rationale": palette.rationale,
                "dmc_codes": palette.dmc_codes,
                "colors": [
                    {
                        "dmc_code": color.dmc_code,
                        "name": color.name,
                        "hex": color.hex,
                        "rgb": color.rgb
                    }
                    for color in colors
                ]
            }
        return info


# Global instance
_fixed_palette_manager: Optional[FixedPaletteManager] = None


def get_fixed_palette_manager() -> FixedPaletteManager:
    """Get global fixed palette manager instance."""
    global _fixed_palette_manager
    if _fixed_palette_manager is None:
        _fixed_palette_manager = FixedPaletteManager()
    return _fixed_palette_manager


def get_fixed_palette(style_name: str) -> FixedPalette:
    """Get fixed palette for specified style."""
    return get_fixed_palette_manager().get_palette(style_name)


def get_fixed_palette_colors(style_name: str) -> List[DMCColor]:
    """Get DMC colors for specified style."""
    return get_fixed_palette_manager().get_palette_colors(style_name)


def list_available_styles() -> List[str]:
    """Get list of available style names."""
    return get_fixed_palette_manager().list_styles()


def get_style_info(style_name: str) -> Dict:
    """Get detailed information about a specific style."""
    return get_fixed_palette_manager().get_style_info()[style_name]


def get_all_styles_info() -> Dict[str, Dict]:
    """Get detailed information about all styles."""
    return get_fixed_palette_manager().get_style_info()
