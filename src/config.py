"""
Configuration management for the paint-by-numbers generator.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any
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

    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return cls(
            canvas=CanvasConfig(**data['canvas']),
            palette=[ColorConfig(**color) for color in data['palette']['colors']],
            processing=ProcessingConfig(**data['processing']),
            pdf=PDFConfig(**data['pdf']),
            symbols=SymbolsConfig(**data['symbols']),
            output=OutputConfig(**data['output'])
        )
    
    def get_color_palette_rgb(self) -> List[List[int]]:
        """Get palette as list of RGB values."""
        return [color.rgb for color in self.palette]
    
    def get_color_palette_names(self) -> List[str]:
        """Get palette as list of color names."""
        return [color.name for color in self.palette]
    
    def get_color_palette_codes(self) -> List[str]:
        """Get palette as list of DMC codes."""
        return [color.dmc_code for color in self.palette]
