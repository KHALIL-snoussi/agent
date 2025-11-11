"""
Configuration management for the paint-by-numbers generator.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class CanvasConfig:
    width_cm: float
    height_cm: float
    drill_size_mm: float
    aspect_ratio: float


@dataclass
class ColorConfig:
    name: str
    rgb: List[int]
    dmc_code: str


@dataclass
class ProcessingConfig:
    dpi: int
    color_space: str
    quantization_method: str
    dithering: bool
    seed: int
    style_adjustments: Optional[Dict[str, float]] = None


@dataclass
class PDFConfig:
    page_size: str
    tiling: bool
    overlap_mm: float
    crop_marks: bool
    margins_mm: float


@dataclass
class SymbolsConfig:
    symbol_set: List[str]
    font_size: int
    min_contrast_ratio: float

# Enhanced symbol sets for better readability
ENHANCED_SYMBOL_SETS = {
    "high_contrast": ["●", "■", "▲", "◆", "★", "✦", "♥"],
    "geometric": ["●", "■", "▲", "◆", "★", "✦", "⬟"],
    "alphanumeric": ["A", "B", "C", "D", "E", "F", "G"],
    "mixed": ["●", "■", "▲", "◆", "★", "✦", "♥"]
}


@dataclass
class OutputConfig:
    spare_percentage: float
    include_instructions: bool
    include_legend: bool
    preview_size: List[int]


@dataclass
class Config:
    canvas: CanvasConfig
    palette: List[ColorConfig]
    processing: ProcessingConfig
    pdf: PDFConfig
    symbols: SymbolsConfig
    output: OutputConfig
    style_preset: Optional[str] = None

    @classmethod
    def from_yaml(cls, config_path: str, style_preset: Optional[str] = None) -> 'Config':
        """Load configuration from YAML file with optional style preset."""
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        config = cls(
            canvas=CanvasConfig(**data['canvas']),
            palette=[ColorConfig(**color) for color in data['palette']['colors']],
            processing=ProcessingConfig(**data['processing']),
            pdf=PDFConfig(**data['pdf']),
            symbols=SymbolsConfig(**data['symbols']),
            output=OutputConfig(**data['output']),
            style_preset=style_preset
        )
        
        # Apply style preset if specified
        if style_preset:
            config.apply_style_preset(style_preset)
        
        return config
    
    def apply_style_preset(self, style_preset: str):
        """Apply a style preset to the configuration."""
        from .style_presets import StyleManager
        
        style_manager = StyleManager()
        preset = style_manager.get_preset(style_preset)
        
        if preset:
            # Update palette with style preset colors
            self.palette = preset.palette
            self.style_preset = style_preset
            
            # Update symbol set if style specifies one
            if hasattr(preset, 'symbol_set') and preset.symbol_set:
                from . import ENHANCED_SYMBOL_SETS
                if preset.symbol_set in ENHANCED_SYMBOL_SETS:
                    self.symbols.symbol_set = ENHANCED_SYMBOL_SETS[preset.symbol_set]
            
            # Update processing settings if style has adjustments
            if hasattr(preset, 'processing') and preset.processing:
                # Add style adjustments to processing config
                if not hasattr(self.processing, 'style_adjustments'):
                    self.processing.style_adjustments = {}
                
                self.processing.style_adjustments = {
                    'contrast_enhancement': preset.processing.get('contrast_enhancement', 1.0),
                    'brightness_adjustment': preset.processing.get('brightness_adjustment', 0.0),
                    'saturation_boost': preset.processing.get('saturation_boost', 0.0)
                }
    
    def get_color_palette_rgb(self) -> List[List[int]]:
        """Get palette as list of RGB values."""
        return [color.rgb for color in self.palette]
    
    def get_color_palette_names(self) -> List[str]:
        """Get palette as list of color names."""
        return [color.name for color in self.palette]
    
    def get_color_palette_codes(self) -> List[str]:
        """Get palette as list of DMC codes."""
        return [color.dmc_code for color in self.palette]
