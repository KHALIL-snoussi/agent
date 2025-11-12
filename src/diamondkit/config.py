"""
Configuration management for diamond painting kit generator.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Literal, Optional
from pathlib import Path


@dataclass
class CanvasConfig:
    """Canvas configuration parameters."""
    width_cm: float = 30.0
    height_cm: float = 40.0
    drill_shape: Literal["square", "round"] = "square"
    drill_size_mm: float = 2.5
    
    def __post_init__(self):
        """Set default round drill size if not specified."""
        if self.drill_shape == "round" and self.drill_size_mm == 2.5:
            self.drill_size_mm = 2.8
    
    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio."""
        return self.width_cm / self.height_cm
    
    @property
    def cells_w(self) -> int:
        """Calculate number of cells horizontally."""
        return int(self.width_cm * 10 / self.drill_size_mm)
    
    @property
    def cells_h(self) -> int:
        """Calculate number of cells vertically."""
        return int(self.height_cm * 10 / self.drill_size_mm)


@dataclass
class PaletteConfig:
    """Palette and color configuration."""
    mode: Literal["dmc", "custom"] = "dmc"
    max_colors: int = 50
    preserve_skin_tones: bool = True
    dmc_file: str = "data/dmc.csv"


@dataclass
class DitherConfig:
    """Dithering configuration."""
    mode: Literal["none", "ordered", "fs"] = "ordered"
    strength: float = 0.35
    auto_disable_flat: bool = True
    variance_threshold: float = 10.0


@dataclass
class ExportConfig:
    """Export configuration."""
    # Tiling settings
    page: Literal["A4", "A3"] = "A4"
    overlap_mm: float = 5.0
    margin_mm: float = 8.0
    
    # Legend settings
    spare_ratio: float = 0.10
    bag_size: int = 200
    
    # Output settings
    pdf_dpi: int = 300
    preview_size: tuple[int, int] = (1200, 1600)


@dataclass
class ProcessingConfig:
    """Image processing configuration."""
    seed: Optional[int] = 42
    color_space: str = "Lab"
    quantization_method: str = "kmeans"


@dataclass
class Config:
    """Main configuration class."""
    # File paths
    input: str = ""
    output_dir: str = "out"
    config_file: Optional[str] = None
    
    # Component configurations
    canvas: CanvasConfig = field(default_factory=CanvasConfig)
    palette: PaletteConfig = field(default_factory=PaletteConfig)
    dither: DitherConfig = field(default_factory=DitherConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str, **overrides) -> "Config":
        """Load configuration from YAML file with optional overrides."""
        if not os.path.exists(config_path):
            # Return default config if file doesn't exist
            config = cls()
            config.config_file = config_path
            return config
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
        except UnicodeDecodeError:
            # Fallback to latin-1 encoding if utf-8 fails
            with open(config_path, 'r', encoding='latin-1') as f:
                data = yaml.safe_load(f) or {}
        
        # Create config with nested structures
        config = cls(
            input=data.get('input', ''),
            output_dir=data.get('output_dir', 'out'),
            config_file=config_path,
            canvas=CanvasConfig(**data.get('canvas', {})),
            palette=PaletteConfig(**data.get('palette', {})),
            dither=DitherConfig(**data.get('dither', {})),
            export=ExportConfig(**data.get('export', {})),
            processing=ProcessingConfig(**data.get('processing', {}))
        )
        
        # Apply CLI overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif hasattr(config.canvas, key):
                setattr(config.canvas, key, value)
            elif hasattr(config.palette, key):
                setattr(config.palette, key, value)
            elif hasattr(config.dither, key):
                setattr(config.dither, key, value)
            elif hasattr(config.export, key):
                setattr(config.export, key, value)
            elif hasattr(config.processing, key):
                setattr(config.processing, key, value)
        
        # Validate configuration
        config.validate()
        
        return config
    
    def validate(self):
        """Validate configuration parameters."""
        if self.canvas.width_cm <= 0 or self.canvas.height_cm <= 0:
            raise ValueError("Canvas dimensions must be positive")
        
        if self.canvas.drill_size_mm <= 0:
            raise ValueError("Drill size must be positive")
        
        if not (0 <= self.dither.strength <= 1):
            raise ValueError("Dither strength must be between 0 and 1")
        
        if self.palette.max_colors < 1:
            raise ValueError("Max colors must be at least 1")
        
        if self.export.spare_ratio < 0:
            raise ValueError("Spare ratio must be non-negative")
        
        if self.export.bag_size < 1:
            raise ValueError("Bag size must be at least 1")
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'input': self.input,
            'output_dir': self.output_dir,
            'canvas': {
                'width_cm': self.canvas.width_cm,
                'height_cm': self.canvas.height_cm,
                'drill_shape': self.canvas.drill_shape,
                'drill_size_mm': self.canvas.drill_size_mm
            },
            'palette': {
                'mode': self.palette.mode,
                'max_colors': self.palette.max_colors,
                'preserve_skin_tones': self.palette.preserve_skin_tones,
                'dmc_file': self.palette.dmc_file
            },
            'dither': {
                'mode': self.dither.mode,
                'strength': self.dither.strength,
                'auto_disable_flat': self.dither.auto_disable_flat,
                'variance_threshold': self.dither.variance_threshold
            },
            'export': {
                'page': self.export.page,
                'overlap_mm': self.export.overlap_mm,
                'margin_mm': self.export.margin_mm,
                'spare_ratio': self.export.spare_ratio,
                'bag_size': self.export.bag_size,
                'pdf_dpi': self.export.pdf_dpi,
                'preview_size': self.export.preview_size
            },
            'processing': {
                'seed': self.processing.seed,
                'color_space': self.processing.color_space,
                'quantization_method': self.processing.quantization_method
            }
        }
    
    def save_yaml(self, path: Optional[str] = None):
        """Save configuration to YAML file."""
        if path is None:
            path = self.config_file or "config.yaml"
        
        # Ensure directory exists
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
