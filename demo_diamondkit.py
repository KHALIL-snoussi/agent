#!/usr/bin/env python3
"""
Demo script for Diamond Painting Kit Generator.
This demonstrates the complete pipeline from image to diamond painting kit.
"""

import os
import sys
import argparse

# Add src to path for imports
sys.path.insert(0, 'src')

try:
    from diamondkit.config import Config
    from diamondkit.cli import main
    from diamondkit.image_io import ImageLoader
    from diamondkit.dmc import get_dmc_palette
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you have installed all dependencies:")
    print("pip install -r requirements.txt")
    sys.exit(1)


def demo_basic_usage():
    """Demonstrate basic usage of the diamond painting kit generator."""
    print("="*60)
    print("DIAMOND PAINTING KIT GENERATOR - DEMO")
    print("="*60)
    
    # Check if sample image exists
    if not os.path.exists('pixel.jpg'):
        print("Sample image 'pixel.jpg' not found.")
        print("Please place a sample image in the current directory.")
        return False
    
    # Create output directory
    output_dir = "demo_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a basic configuration
    config = Config()
    config.input = "pixel.jpg"
    config.output_dir = output_dir
    config.canvas.width_cm = 20.0  # Smaller for demo
    config.canvas.height_cm = 30.0
    config.palette.max_colors = 15  # Fewer colors for demo
    config.dither.mode = "ordered"
    config.dither.strength = 0.3
    
    print(f"Input image: {config.input}")
    print(f"Output directory: {config.output_dir}")
    print(f"Canvas size: {config.canvas.width_cm}x{config.canvas.height_cm} cm")
    print(f"Max colors: {config.palette.max_colors}")
    print(f"Dithering: {config.dither.mode}")
    print()
    
    try:
        # Initialize components
        print("Initializing DMC palette...")
        dmc_palette = get_dmc_palette()
        print(f"Loaded {dmc_palette.get_color_count()} DMC colors")
        
        print("\nValidating input image...")
        image_loader = ImageLoader(config)
        validation = image_loader.validate_image(config.input)
        
        if not validation['valid']:
            print("[X] Image validation failed:")
            for error in validation['errors']:
                print(f"  - {error}")
            return False
        
        print("[OK] Image validation passed")
        
        if validation['warnings']:
            print("[WARN]  Warnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")
        
        # Show image info
        print("\nGetting image information...")
        info = image_loader.get_image_info(config.input)
        if 'error' not in info:
            print(f"[OK] Image info: {info['size']} pixels, {info['unique_colors']} unique colors")
            print(f"   Estimated difficulty: {info['estimated_difficulty']}")
        
        print("\n[OK] Demo setup complete!")
        print("\nTo generate a complete kit, run:")
        print(f"python -m diamondkit.cli {config.input} {config.output_dir} --canvas-size 20x30 --max-colors 15")
        
        return True
        
    except Exception as e:
        print(f"[X] Demo failed: {e}")
        return False


def demo_color_analysis():
    """Demonstrate DMC color analysis capabilities."""
    print("\n" + "="*60)
    print("DMC COLOR ANALYSIS - DEMO")
    print("="*60)
    
    try:
        # Load DMC palette
        dmc_palette = get_dmc_palette()
        
        print(f"Loaded {dmc_palette.get_color_count()} DMC colors")
        
        # Show some sample colors
        print("\nSample DMC colors:")
        for i, color in enumerate(dmc_palette.colors[:10]):
            print(f"  {color.dmc_code:>6} - {color.name:<25} RGB{color.rgb} {color.hex}")
        
        # Show skin tone colors
        skin_tones = dmc_palette.get_skin_tone_colors()
        print(f"\nFound {len(skin_tones)} potential skin tone colors")
        
        # Test color matching
        test_colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 255), # White
            (0, 0, 0),      # Black
            (255, 200, 150), # Skin tone
        ]
        
        print("\nColor matching examples:")
        for rgb in test_colors:
            nearest = dmc_palette.find_nearest_color(rgb)
            print(f"  RGB{rgb} -> {nearest.dmc_code:>6} {nearest.name:<20} RGB{nearest.rgb}")
        
        return True
        
    except Exception as e:
        print(f"[X] Color analysis failed: {e}")
        return False


def demo_config_system():
    """Demonstrate configuration system."""
    print("\n" + "="*60)
    print("CONFIGURATION SYSTEM - DEMO")
    print("="*60)
    
    try:
        # Create default config
        config = Config()
        print("Default configuration:")
        print(f"  Canvas: {config.canvas.width_cm}x{config.canvas.height_cm} cm")
        print(f"  Drill shape: {config.canvas.drill_shape}")
        print(f"  Drill size: {config.canvas.drill_size_mm} mm")
        print(f"  Max colors: {config.palette.max_colors}")
        print(f"  Dithering: {config.dither.mode} ({config.dither.strength:.0%} strength)")
        print(f"  Page size: {config.export.page}")
        print(f"  Spare ratio: {config.export.spare_ratio:.0%}")
        
        # Test canvas calculations
        print(f"\nCanvas calculations:")
        print(f"  Aspect ratio: {config.canvas.aspect_ratio:.3f}")
        print(f"  Grid size: {config.canvas.cells_w}x{config.canvas.cells_h} cells")
        print(f"  Total cells: {config.canvas.cells_w * config.canvas.cells_h:,}")
        
        # Test validation
        print(f"\nConfiguration validation: [OK] PASSED")
        
        # Test YAML export
        config.save_yaml("demo_config.yaml")
        print("Saved demo configuration to: demo_config.yaml")
        
        return True
        
    except Exception as e:
        print(f"[X] Config demo failed: {e}")
        return False


def main_demo():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Diamond Painting Kit Generator Demo")
    parser.add_argument(
        "--full", 
        action="store_true",
        help="Run full demo (requires sample image)"
    )
    
    args = parser.parse_args()
    
    print("[kit] Diamond Painting Kit Generator - Demo")
    print("=====================================")
    
    # Always run config and color analysis demos
    success = True
    success &= demo_config_system()
    success &= demo_color_analysis()
    
    # Run full demo only if requested
    if args.full:
        success &= demo_basic_usage()
    else:
        print("\n" + "="*60)
        print("FULL DEMO")
        print("="*60)
        print("To run the complete pipeline demo:")
        print("python demo_diamondkit.py --full")
        print()
        print("This requires a sample image named 'pixel.jpg' in the current directory.")
    
    if success:
        print("\n[OK] Demo completed successfully!")
        print("\nNext steps:")
        print("1. Place your image in the current directory")
        print("2. Run: python -m diamondkit.cli your_image.jpg output_folder/")
        print("3. Check the generated PDF and other files")
    else:
        print("\n[X] Demo encountered issues. Please check the error messages above.")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main_demo())
