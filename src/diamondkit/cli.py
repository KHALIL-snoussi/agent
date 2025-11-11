"""
Command-line interface for diamond painting kit generator.
"""

import os
import sys
import argparse
from pathlib import Path

from diamondkit.config import Config
from diamondkit.image_io import ImageLoader
from diamondkit.dmc import get_dmc_palette
from diamondkit.quantize import ColorQuantizer
from diamondkit.dither import DitherEngine
from diamondkit.grid import CanvasGrid
from diamondkit.preview import PreviewGenerator
from diamondkit.export import ExportManager


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate professional DMC diamond painting kits from images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a basic kit
  python -m diamondkit.cli input.jpg output/
  
  # Custom canvas size and colors
  python -m diamondkit.cli input.jpg output/ --canvas-size 40x50 --max-colors 60
  
  # Use round drills and Floyd-Steinberg dithering
  python -m diamondkit.cli input.jpg output/ --drill-shape round --dither fs
  
  # Use configuration file
  python -m diamondkit.cli input.jpg output/ --config my_config.yaml
        """
    )
    
    # Input/Output
    parser.add_argument(
        "input",
        help="Input image file (JPG/PNG)"
    )
    parser.add_argument(
        "output",
        help="Output directory for generated kit"
    )
    
    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="YAML configuration file path"
    )
    
    # Canvas settings
    parser.add_argument(
        "--canvas-size",
        type=str,
        default="30x40",
        help="Canvas size in cm (default: 30x40)"
    )
    parser.add_argument(
        "--drill-shape",
        choices=["square", "round"],
        default="square",
        help="Drill shape (default: square)"
    )
    parser.add_argument(
        "--drill-size-mm",
        type=float,
        help="Drill size in mm (default: 2.5 for square, 2.8 for round)"
    )
    
    # Palette settings
    parser.add_argument(
        "--max-colors",
        type=int,
        default=50,
        help="Maximum number of DMC colors (default: 50)"
    )
    parser.add_argument(
        "--preserve-skin-tones",
        action="store_true",
        default=True,
        help="Preserve skin tones in quantization"
    )
    parser.add_argument(
        "--no-preserve-skin-tones",
        action="store_false",
        dest="preserve_skin_tones",
        help="Disable skin tone preservation"
    )
    
    # Dithering settings
    parser.add_argument(
        "--dither",
        choices=["none", "ordered", "fs"],
        default="ordered",
        help="Dithering mode (default: ordered)"
    )
    parser.add_argument(
        "--dither-strength",
        type=float,
        default=0.35,
        help="Dithering strength for ordered mode 0-1 (default: 0.35)"
    )
    
    # Processing settings
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible results (default: 42)"
    )
    
    # Export settings
    parser.add_argument(
        "--page-size",
        choices=["A4", "A3"],
        default="A4",
        help="PDF page size (default: A4)"
    )
    parser.add_argument(
        "--spare-ratio",
        type=float,
        default=0.10,
        help="Spare drill ratio 0-1 (default: 0.10)"
    )
    parser.add_argument(
        "--bag-size",
        type=int,
        default=200,
        help="Drills per bag (default: 200)"
    )
    
    # Utility commands
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate input image, don't generate kit"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show detailed image information"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser


def parse_canvas_size(size_str: str) -> tuple[float, float]:
    """Parse canvas size string like '30x40' into (30.0, 40.0)."""
    try:
        parts = size_str.lower().split('x')
        if len(parts) != 2:
            raise ValueError("Format must be WxH (e.g., 30x40)")
        
        width = float(parts[0])
        height = float(parts[1])
        
        if width <= 0 or height <= 0:
            raise ValueError("Dimensions must be positive")
        
        return width, height
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid canvas size '{size_str}': {e}")


def validate_arguments(args: argparse.Namespace) -> bool:
    """Validate command-line arguments."""
    errors = []
    
    # Check input file
    if not os.path.exists(args.input):
        errors.append(f"Input file not found: {args.input}")
    
    # Check canvas size format
    try:
        width, height = parse_canvas_size(args.canvas_size)
    except ValueError as e:
        errors.append(str(e))
    
    # Check ranges
    if not (0 <= args.dither_strength <= 1):
        errors.append("Dither strength must be between 0 and 1")
    
    if args.max_colors < 1:
        errors.append("Max colors must be at least 1")
    
    if args.max_colors > 100:
        errors.append("Max colors should not exceed 100 for practical use")
    
    if args.spare_ratio < 0:
        errors.append("Spare ratio must be non-negative")
    
    if args.bag_size < 1:
        errors.append("Bag size must be at least 1")
    
    if args.drill_size_mm and args.drill_size_mm <= 0:
        errors.append("Drill size must be positive")
    
    # Report errors
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


def load_config(args: argparse.Namespace) -> Config:
    """Load configuration from arguments and optional config file."""
    # Parse canvas size
    width_cm, height_cm = parse_canvas_size(args.canvas_size)
    
    # Create overrides dictionary
    overrides = {
        'input': args.input,
        'output_dir': args.output,
        'canvas': {
            'width_cm': width_cm,
            'height_cm': height_cm,
            'drill_shape': args.drill_shape
        },
        'palette': {
            'max_colors': args.max_colors,
            'preserve_skin_tones': args.preserve_skin_tones
        },
        'dither': {
            'mode': args.dither,
            'strength': args.dither_strength
        },
        'processing': {
            'seed': args.seed
        },
        'export': {
            'page': args.page_size,
            'spare_ratio': args.spare_ratio,
            'bag_size': args.bag_size
        }
    }
    
    # Add drill size if specified
    if args.drill_size_mm:
        overrides['canvas']['drill_size_mm'] = args.drill_size_mm
    
    # Load configuration
    config_path = args.config or "config.yaml"
    config = Config.from_yaml(config_path, **overrides)
    
    return config


def show_image_info(image_path: str, config: Config):
    """Show detailed information about input image."""
    loader = ImageLoader(config)
    info = loader.get_image_info(image_path)
    
    if 'error' in info:
        print(f"Error: {info['error']}")
        return
    
    print("\n" + "="*60)
    print("IMAGE INFORMATION")
    print("="*60)
    print(f"Filename: {info['filename']}")
    print(f"Size: {info['size'][0]} × {info['size'][1]} pixels")
    print(f"Aspect ratio: {info['aspect_ratio']:.3f}")
    print(f"Mode: {info['mode']}")
    print(f"Format: {info['format']}")
    print(f"File size: {info['file_size'] / 1024:.1f} KB")
    print(f"Unique colors: {info['unique_colors']:,}")
    print(f"Estimated difficulty: {info['estimated_difficulty']}")
    print(f"Mean color: RGB{info['mean_color']}")
    print(f"Color std: RGB{info['std_color']}")
    
    # Canvas compatibility
    canvas_aspect = config.canvas.aspect_ratio
    current_aspect = info['aspect_ratio']
    aspect_diff = abs(current_aspect - canvas_aspect) / canvas_aspect
    
    print(f"\nCanvas compatibility:")
    print(f"  Target aspect ratio: {canvas_aspect:.3f}")
    print(f"  Current aspect ratio: {current_aspect:.3f}")
    print(f"  Difference: {aspect_diff:.1%}")
    
    if aspect_diff < 0.05:
        print("  ✓ Excellent match")
    elif aspect_diff < 0.15:
        print("  ⚠ Good match")
    elif aspect_diff < 0.30:
        print("  ⚠ Moderate adjustment needed")
    else:
        print("  ⚠ Significant adjustment required")
    
    print("="*60)


def validate_input_image(image_path: str, config: Config) -> bool:
    """Validate input image for processing."""
    loader = ImageLoader(config)
    validation = loader.validate_image(image_path)
    
    print(f"\nValidating: {image_path}")
    
    if validation['valid']:
        print("✓ Image validation passed")
        
        # Show warnings
        if validation['warnings']:
            print("\nWarnings:")
            for warning in validation['warnings']:
                print(f"  ⚠ {warning}")
        
        return True
    else:
        print("✗ Image validation failed")
        print("\nErrors:")
        for error in validation['errors']:
            print(f"  ✗ {error}")
        
        return False


def generate_kit(config: Config) -> bool:
    """Generate complete diamond painting kit."""
    try:
        print("\n" + "="*60)
        print("DIAMOND PAINTING KIT GENERATOR")
        print("="*60)
        
        # Initialize components
        print("Initializing components...")
        dmc_palette = get_dmc_palette(config.palette.dmc_file)
        image_loader = ImageLoader(config)
        quantizer = ColorQuantizer(config, dmc_palette)
        dither_engine = DitherEngine(config)
        
        # Load and process image
        print("\n" + "-"*40)
        print("IMAGE PROCESSING")
        print("-"*40)
        image_lab, metadata = image_loader.load_image(config.input)
        
        # Quantize colors
        print("\n" + "-"*40)
        print("COLOR QUANTIZATION")
        print("-"*40)
        quantized_lab, dmc_colors = quantizer.quantize_image(image_lab)
        
        # Apply dithering
        print("\n" + "-"*40)
        print("DITHERING")
        print("-"*40)
        dithered_lab = dither_engine.apply_dithering(image_lab, quantized_lab)
        
        # Create grid
        print("\n" + "-"*40)
        print("GRID GENERATION")
        print("-"*40)
        canvas_grid = CanvasGrid(config, dmc_colors)
        grid_data = canvas_grid.create_grid(dithered_lab)
        
        # Generate outputs
        print("\n" + "-"*40)
        print("EXPORT GENERATION")
        print("-"*40)
        preview_generator = PreviewGenerator(config)
        export_manager = ExportManager(config)
        
        # Create preview
        preview_image = preview_generator.create_preview(
            dithered_lab, dmc_colors, canvas_grid
        )
        
        # Export all files
        export_manager.export_complete_kit(
            canvas_grid, preview_image, metadata
        )
        
        print("\n" + "="*60)
        print("✓ KIT GENERATION COMPLETE")
        print("="*60)
        print(f"Output directory: {config.output_dir}")
        print(f"DMC colors used: {len(dmc_colors)}")
        print(f"Grid size: {canvas_grid.cells_w} × {canvas_grid.cells_h}")
        print(f"Total drills: {canvas_grid._get_total_drills():,}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during kit generation: {e}")
        if '--verbose' in sys.argv:
            import traceback
            traceback.print_exc()
        return False


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)
    
    # Load configuration
    try:
        config = load_config(args)
    except Exception as e:
        print(f"Configuration error: {e}")
        print("Using default configuration...")
        # Create default config with command line overrides
        config = Config()
        if args.input:
            config.input = args.input
        if args.output:
            config.output_dir = args.output
        # Apply other overrides manually
        try:
            width_cm, height_cm = parse_canvas_size(args.canvas_size)
            config.canvas.width_cm = width_cm
            config.canvas.height_cm = height_cm
            config.canvas.drill_shape = args.drill_shape
            config.palette.max_colors = args.max_colors
            config.palette.preserve_skin_tones = args.preserve_skin_tones
            config.dither.mode = args.dither
            config.dither.strength = args.dither_strength
            config.processing.seed = args.seed
            config.export.page = args.page_size
            config.export.spare_ratio = args.spare_ratio
            config.export.bag_size = args.bag_size
            if args.drill_size_mm:
                config.canvas.drill_size_mm = args.drill_size_mm
        except Exception as override_error:
            print(f"Warning: Could not apply all overrides: {override_error}")
    
    # Handle utility commands
    if args.info:
        show_image_info(args.input, config)
        return
    
    if args.validate_only:
        if validate_input_image(args.input, config):
            print("✓ Image is ready for processing")
            sys.exit(0)
        else:
            print("✗ Image validation failed")
            sys.exit(1)
    
    # Generate kit
    success = generate_kit(config)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
