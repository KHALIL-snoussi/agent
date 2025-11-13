"""
Command-line interface for the paint-by-numbers generator.
"""

import click
import os
import sys
from pathlib import Path

from .paint_generator import PaintGenerator


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    Paint-by-Numbers / Diamond-Painting Generator
    
    Convert any photo into a print-ready paint-by-numbers or diamond-painting kit.
    """
    pass


@cli.command()
@click.argument('input_image', type=click.Path(exists=True))
@click.argument('output_pdf', type=click.Path())
@click.option('--config', '-c', default='config.yaml', help='Configuration file path')
@click.option('--style', '-st', type=click.Choice(['vintage', 'minimalist', 'vibrant', 'pastel', 'artistic', 'monochrome', 'popart']), 
              help='Apply a style preset (overrides default palette)')
@click.option('--save-intermediate', '-s', is_flag=True, help='Save intermediate processing images')
@click.option('--palette-info', '-p', is_flag=True, help='Show current palette information')
@click.option('--quality-check', '-q', is_flag=True, help='Run quality assessment before processing')
def generate(input_image, output_pdf, config, style, save_intermediate, palette_info, quality_check):
    """
    Generate a paint-by-numbers kit from an image.
    
    INPUT_IMAGE: Path to input image (JPG/PNG supported)
    OUTPUT_PDF: Path for output PDF file
    """
    try:
        # Initialize generator with optional style preset
        generator = PaintGenerator(config, style_preset=style)
        
        # Show palette info if requested
        if palette_info:
            generator.get_palette_info()
            return
        
        # Run quality check if requested
        if quality_check:
            click.echo("[search] Running quality assessment...")
            quality_result = generator.assess_image_quality(input_image)
            report = generator.quality_assessor.generate_quality_report(quality_result)
            click.echo(report)
            
            if quality_result.level.value == "Low":
                if not click.confirm("[WARN]  Image quality is low. Continue anyway?"):
                    click.echo("Processing cancelled.")
                    return
        
        # Validate input image
        generator.validate_image(input_image)
        
        # Ensure output directory exists
        output_path = Path(output_pdf)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate the kit
        click.echo(f"[kit] Generating paint-by-numbers kit...")
        if style:
            click.echo(f"[clipboard] Using style preset: {style}")
        
        retrieval_code = generator.generate_kit(input_image, output_pdf, save_intermediate)
        
        if retrieval_code:
            click.echo(f"[key] Retrieval code: {retrieval_code}")
            click.echo("[save] Keep this code safe to reprint your kit anytime!")
        
        click.echo(f"[OK] Kit generated successfully: {output_pdf}")
        
    except Exception as e:
        click.echo(f"[X] Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', default='config.yaml', help='Configuration file path')
def palette(config):
    """Display the current color palette information."""
    try:
        generator = PaintGenerator(config)
        generator.get_palette_info()
    except Exception as e:
        click.echo(f"[X] Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_image', type=click.Path(exists=True))
@click.option('--config', '-c', default='config.yaml', help='Configuration file path')
def validate(input_image, config):
    """Validate an input image for processing."""
    try:
        generator = PaintGenerator(config)
        if generator.validate_image(input_image):
            click.echo("[OK] Image is valid for processing")
    except Exception as e:
        click.echo(f"[X] Validation failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--output', '-o', default='config.yaml', help='Output configuration file path')
def init_config(output):
    """Create a default configuration file."""
    try:
        # Check if config already exists
        if os.path.exists(output) and not click.confirm(f"Configuration file '{output}' already exists. Overwrite?"):
            click.echo("Configuration creation cancelled.")
            return
        
        # Copy default config content
        default_config = """# Paint-by-Numbers / Diamond-Painting Generator Configuration

# Canvas settings
canvas:
  width_cm: 30  # Target canvas width in cm
  height_cm: 40  # Target canvas height in cm
  drill_size_mm: 2.5  # Size of each drill/dot in mm
  aspect_ratio: 0.75  # 3:4 aspect ratio (width/height)

# Color palette - 7 colors with DMC-equivalent codes
palette:
  colors:
    - name: "Black"
      rgb: [0, 0, 0]
      dmc_code: "310"
    - name: "White"
      rgb: [255, 255, 255]
      dmc_code: "Blanc"
    - name: "Red"
      rgb: [220, 20, 60]
      dmc_code: "666"
    - name: "Blue"
      rgb: [0, 100, 200]
      dmc_code: "791"
    - name: "Yellow"
      rgb: [255, 215, 0]
      dmc_code: "742"
    - name: "Green"
      rgb: [34, 139, 34]
      dmc_code: "704"
    - name: "Purple"
      rgb: [147, 112, 219]
      dmc_code: "554"

# Image processing settings
processing:
  dpi: 300  # Output DPI for PDF
  color_space: "Lab"  # Use Lab color space for quantization
  quantization_method: "kmeans"  # kmeans or median_cut
  dithering: true  # Apply error diffusion dithering
  seed: 42  # Random seed for reproducible results

# PDF generation settings
pdf:
  page_size: "A4"  # A4 or A3
  tiling: true  # Split large pattern across multiple pages
  overlap_mm: 5  # Overlap between tiled pages in mm
  crop_marks: true  # Add crop marks for alignment
  margins_mm: 10  # Page margins in mm

# Symbol settings
symbols:
  # High-contrast symbols for grid
  symbol_set: ["*", "#", "^", "*", "*", "*", "*"]
  font_size: 8  # Symbol font size in points
  min_contrast_ratio: 4.5  # Minimum contrast ratio for readability

# Output settings
output:
  spare_percentage: 0.15  # Add 15% extra drills/paint
  include_instructions: true  # Include assembly instructions
  include_legend: true  # Include color legend
  preview_size: [200, 267]  # Preview image size in pixels
"""
        
        with open(output, 'w') as f:
            f.write(default_config)
        
        click.echo(f"[OK] Default configuration created: {output}")
        
    except Exception as e:
        click.echo(f"[X] Error creating configuration: {e}", err=True)
        sys.exit(1)


@cli.command()
def demo():
    """Create a demo kit using a sample image."""
    try:
        # Generate a sample image
        click.echo("Creating demo image...")
        demo_image_path = create_demo_image()
        
        # Generate the kit
        output_path = "demo_kit.pdf"
        generator = PaintGenerator()
        generator.generate_kit(demo_image_path, output_path, save_intermediate=True)
        
        click.echo(f"[OK] Demo kit created: {output_path}")
        click.echo(f"[folder] Demo image: {demo_image_path}")
        
    except Exception as e:
        click.echo(f"[X] Error creating demo: {e}", err=True)
        sys.exit(1)


def create_demo_image():
    """Create a simple demo image for testing."""
    import numpy as np
    from PIL import Image, ImageDraw
    
    # Create a colorful test pattern
    width, height = 400, 533  # 3:4 aspect ratio
    
    # Create gradient background
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    
    # Create a simple geometric pattern
    for y in range(0, height, 40):
        for x in range(0, width, 40):
            color = (
                (x * 255) // width,
                (y * 255) // height,
                ((x + y) * 255) // (width + height)
            )
            draw.rectangle([x, y, x+35, y+35], fill=color)
    
    # Add some circles
    for i in range(5):
        x = (i + 1) * width // 6
        y = height // 2
        radius = 30
        color = (
            (i * 255) // 4,
            255 - (i * 255) // 4,
            128
        )
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
    
    demo_path = "demo_image.png"
    img.save(demo_path)
    return demo_path


if __name__ == '__main__':
    cli()
