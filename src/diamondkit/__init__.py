"""
Diamond Painting Kit Generator

A professional tool for converting images into DMC-compatible diamond painting kits
with full color palette support, dithering options, and production-quality outputs.
"""

__version__ = "1.0.0"
__author__ = "Diamond Kit Generator"

from .config import Config
from .dmc import DMCPalette
from .image_io import ImageLoader
from .quantize import ColorQuantizer
from .dither import DitherEngine
from .grid import CanvasGrid
from .pdf import QBRIXPDFGenerator as PDFGenerator
from .preview import PreviewGenerator
from .export import ExportManager
from . import cli

__all__ = [
    "Config",
    "DMCPalette", 
    "ImageLoader",
    "ColorQuantizer",
    "DitherEngine",
    "CanvasGrid",
    "PDFGenerator",
    "PreviewGenerator",
    "ExportManager",
    "cli"
]
