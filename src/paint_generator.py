"""
Main paint-by-numbers generator class that coordinates all components.
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional

from .config import Config
from .image_processor import ImageProcessor
from .pdf_generator import PDFGenerator
from .quality_assessor import QualityAssessor
from .kit_manifest import ManifestManager


class PaintGenerator:
    """Main generator class that coordinates entire pipeline."""
    
    def __init__(self, config_path: str = "config.yaml", style_preset: Optional[str] = None):
        """
        Initialize paint generator.
        
        Args:
            config_path: Path to configuration file
            style_preset: Optional style preset to apply
        """
        self.config = Config.from_yaml(config_path, style_preset)
        self.image_processor = ImageProcessor(self.config)
        self.pdf_generator = PDFGenerator(self.config)
        self.quality_assessor = QualityAssessor()
        self.manifest_manager = ManifestManager()
    
    def generate_kit(self, input_image_path: str, output_pdf_path: str, 
                    save_intermediate: bool = False) -> str:
        """
        Generate complete paint-by-numbers kit from image.
        
        Args:
            input_image_path: Path to input image
            output_pdf_path: Path for output PDF
            save_intermediate: Whether to save intermediate images
            
        Returns:
            Path to generated PDF
        """
        print(f"Processing image: {input_image_path}")
        
        # Step 1: Preprocess image
        print("Preprocessing image...")
        preprocessed_image = self.image_processor.preprocess_image(input_image_path)
        
        if save_intermediate:
            self._save_intermediate_image(preprocessed_image, output_pdf_path, "_preprocessed")
        
        # Step 2: Quantize colors
        print("Quantizing colors to palette...")
        quantized_image, color_indices = self.image_processor.quantize_colors(preprocessed_image)
        
        if save_intermediate:
            self._save_intermediate_image(quantized_image, output_pdf_path, "_quantized")
        
        # Step 3: Generate symbol grid
        print("Generating symbol grid...")
        symbol_grid, symbols = self.image_processor.generate_grid_with_symbols(color_indices)
        
        # Step 4: Count color usage
        print("Calculating color quantities...")
        color_counts = self.image_processor.count_color_usage(color_indices)
        
        # Print statistics
        self._print_statistics(color_counts, quantized_image.shape)
        
        # Step 5: Generate retrieval code and manifest
        print("Creating kit manifest...")
        retrieval_code = self.manifest_manager.generate_retrieval_code(
            input_image_path,
            self.config.style_preset,
            self.config.processing.__dict__,
            {color.name: count for color, count in zip(self.config.palette, color_counts)},
            self.config.canvas.__dict__
        )
        
        # Step 6: Generate PDF
        print("Generating PDF kit...")
        self.pdf_generator.generate_pdf_kit(
            output_pdf_path, 
            quantized_image, 
            symbol_grid, 
            color_counts,
            input_image_path,
            retrieval_code
        )
        
        # Update manifest with PDF path
        self.manifest_manager.update_manifest_pdf_path(retrieval_code, output_pdf_path)
        
        print(f"✅ Kit generated successfully: {output_pdf_path}")
        return retrieval_code
    
    def _save_intermediate_image(self, image: np.ndarray, output_path: str, suffix: str):
        """Save intermediate processing results."""
        base_path = Path(output_path).with_suffix('')
        intermediate_path = f"{base_path}{suffix}.png"
        
        from PIL import Image
        img = Image.fromarray(image)
        img.save(intermediate_path)
        print(f"Saved intermediate image: {intermediate_path}")
    
    def _print_statistics(self, color_counts: list[int], image_shape: tuple[int, int]):
        """Print processing statistics."""
        total_cells = image_shape[0] * image_shape[1]
        total_drills = sum(color_counts)
        
        print("\n" + "="*50)
        print("PROCESSING STATISTICS")
        print("="*50)
        print(f"Image size: {image_shape[1]} × {image_shape[0]} drills")
        print(f"Total cells: {total_cells:,}")
        print(f"Colors used: {len(self.config.palette)}")
        print(f"Spare percentage: {int(self.config.output.spare_percentage * 100)}%")
        print(f"Total drills with spare: {total_drills:,}")
        print("\nColor breakdown:")
        
        for i, (color_config, count) in enumerate(zip(self.config.palette, color_counts)):
            percentage = (count / total_drills) * 100
            print(f"  {color_config.name:8} ({color_config.dmc_code:5}): {count:4,} drills ({percentage:5.1f}%)")
        
        print("="*50 + "\n")
    
    def get_palette_info(self):
        """Print information about the current color palette."""
        print("Current Color Palette:")
        print("-" * 40)
        for i, color in enumerate(self.config.palette):
            rgb_str = f"({color.rgb[0]:3d}, {color.rgb[1]:3d}, {color.rgb[2]:3d})"
            print(f"{i+1}. {color.name:8} - RGB: {rgb_str} - DMC: {color.dmc_code}")
    
    def validate_image(self, image_path: str) -> bool:
        """
        Validate input image requirements.
        
        Args:
            image_path: Path to image file
            
        Returns:
            True if valid, raises exception otherwise
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Check file extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        file_ext = Path(image_path).suffix.lower()
        
        if file_ext not in valid_extensions:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: {valid_extensions}")
        
        # Try to open image
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                # Check minimum size
                if img.size[0] < 100 or img.size[1] < 100:
                    raise ValueError("Image too small. Minimum size: 100×100 pixels")
                
                # Check if image has sufficient resolution
                target_resolution = self.image_processor._calculate_target_resolution()
                if img.size[0] < target_resolution[0] or img.size[1] < target_resolution[1]:
                    print(f"⚠️  Warning: Input image ({img.size[0]}×{img.size[1]}) is smaller than target resolution ({target_resolution[0]}×{target_resolution[1]}). Quality may be reduced.")
        
        except Exception as e:
            raise ValueError(f"Cannot process image: {e}")
        
        return True
    
    def assess_image_quality(self, image_path: str):
        """
        Assess image quality for paint-by-numbers conversion.
        
        Args:
            image_path: Path to image file
            
        Returns:
            QualityAssessment object with detailed analysis
        """
        # Load and preprocess image for quality assessment
        from PIL import Image
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Get a quick quantized version for assessment
        preprocessed = self.image_processor.preprocess_image(image_path)
        quantized, _ = self.image_processor.quantize_colors(preprocessed)
        
        # Assess quality
        assessment = self.quality_assessor.assess_quality(
            img_array, quantized, len(self.config.palette)
        )
        
        return assessment
