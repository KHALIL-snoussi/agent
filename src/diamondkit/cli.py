"""
Command-line interface for QBRIX-quality diamond painting kit generator.
Fixed 7-color DMC palettes with 10,000 cell constraints.
"""

import os
import sys
import argparse
from pathlib import Path

from .kit_generator import generate_diamond_kit, get_available_styles, get_style_info
from .print_math import PrintSpecs


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser for fixed palette system."""
    parser = argparse.ArgumentParser(
        description="Generate QBRIX-quality diamond painting kits with fixed 7-color DMC palettes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
AVAILABLE STYLES:
  ORIGINAL - Balanced palette with primary colors and neutrals
  VINTAGE  - Muted sepia/cream range for traditional looks  
  POPART   - Bold high-contrast colors for vibrant designs

Examples:
  # Generate ORIGINAL style kit
  python -m diamondkit.cli input.jpg output/ --style original
  
  # Generate VINTAGE style kit with custom DPI
  python -m diamondkit.cli input.jpg output/ --style vintage --dpi 600
  
  # Generate POPART style kit with custom cell size
  python -m diamondkit.cli input.jpg output/ --style popart --cell-size-mm 2.5
  
  # List available styles and their palettes
  python -m diamondkit.cli --list-styles
  
  # Get detailed info about a specific style
  python -m diamondkit.cli --style-info vintage
        """
    )
    
    # Mode selection
    parser.add_argument(
        "input",
        nargs="?",
        help="Input image file (JPG/PNG) - required for kit generation"
    )
    parser.add_argument(
        "output",
        nargs="?", 
        help="Output directory for generated kit - required for kit generation"
    )
    
    # Style selection (REQUIRED)
    parser.add_argument(
        "--style", "-s",
        choices=["original", "vintage", "popart"],
        help="Style choice: ORIGINAL, VINTAGE, or POPART (required for kit generation)"
    )
    
    # Print specifications
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Print DPI (>=300 required, default: 600)"
    )
    parser.add_argument(
        "--margin-mm",
        type=float,
        default=12.0,
        help="Paper margins in mm (10-15mm range, default: 12)"
    )
    parser.add_argument(
        "--cell-size-mm",
        type=float,
        default=2.8,
        help="Cell size in mm (2.3-3.0mm range, default: 2.8)"
    )
    
    # Cropping options
    parser.add_argument(
        "--crop",
        type=str,
        help="Crop rectangle as 'x,y,width,height' in normalized coordinates (0-1)"
    )
    
    # Utility commands
    parser.add_argument(
        "--list-styles",
        action="store_true",
        help="List available styles and their DMC palettes"
    )
    parser.add_argument(
        "--style-info",
        type=str,
        choices=["original", "vintage", "popart"],
        help="Show detailed information about a specific style"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate input image and show analysis, don't generate kit"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser


def validate_arguments(args: argparse.Namespace) -> bool:
    """Validate command-line arguments for fixed palette system."""
    # Handle utility commands first
    if args.list_styles or args.style_info:
        return True
    
    # Check required arguments for kit generation
    if not args.input:
        print("Error: Input image file is required for kit generation")
        return False
    
    if not args.output:
        print("Error: Output directory is required for kit generation")
        return False
    
    if not args.style:
        print("Error: Style selection is required. Use --style original|vintage|popart")
        return False
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return False
    
    # Validate print specifications
    if args.dpi < 300:
        print("Error: DPI must be >= 300 for print quality")
        return False
    
    if not (10 <= args.margin_mm <= 15):
        print("Error: Margins must be between 10-15mm")
        return False
    
    if not (2.3 <= args.cell_size_mm <= 3.0):
        print("Error: Cell size must be between 2.3-3.0mm")
        return False
    
    # Parse and validate crop rectangle if provided
    if args.crop:
        try:
            parts = args.crop.split(',')
            if len(parts) != 4:
                raise ValueError("Crop must have 4 values")
            
            x, y, w, h = map(float, parts)
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                raise ValueError("Crop values must be between 0-1")
            if x + w > 1 or y + h > 1:
                raise ValueError("Crop rectangle exceeds image bounds")
                
        except ValueError as e:
            print(f"Error: Invalid crop format '{args.crop}': {e}")
            return False
    
    return True


def parse_crop_rectangle(crop_str: str) -> tuple:
    """Parse crop rectangle string into tuple."""
    parts = crop_str.split(',')
    return tuple(map(float, parts))


def list_styles():
    """List all available styles with their DMC palettes."""
    print("\n" + "="*60)
    print("AVAILABLE STYLES")
    print("="*60)
    
    styles = get_available_styles()
    for style in styles:
        info = get_style_info(style)
        print(f"\n{style.upper()}:")
        print(f"  Description: {info['description']}")
        print(f"  Rationale: {info['rationale']}")
        print(f"  DMC Codes: {', '.join(info['dmc_codes'])}")
        print("  Colors:")
        for color in info['colors']:
            print(f"    {color['dmc_code']}: {color['name']} ({color['hex']})")
    
    print("\n" + "="*60)


def show_style_info(style_name: str):
    """Show detailed information about a specific style."""
    print(f"\n" + "="*60)
    print(f"STYLE: {style_name.upper()}")
    print("="*60)
    
    info = get_style_info(style_name)
    print(f"Description: {info['description']}")
    print(f"Rationale: {info['rationale']}")
    print(f"\nDMC Palette (7 fixed colors):")
    print("-" * 40)
    
    for i, color in enumerate(info['colors']):
        print(f"{i+1}. {color['dmc_code']}: {color['name']}")
        print(f"   RGB: {color['rgb']}")
        print(f"   Hex: {color['hex']}")
        print()
    
    print("="*60)


def generate_kit(args: argparse.Namespace) -> bool:
    """Generate complete diamond painting kit with fixed palette."""
    try:
        # Parse crop rectangle if provided
        crop_rect = None
        if args.crop:
            crop_rect = parse_crop_rectangle(args.crop)
        
        print("\n" + "="*60)
        print("QBRIX DIAMOND PAINTING KIT GENERATOR")
        print("="*60)
        print(f"Style: {args.style.upper()}")
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print(f"DPI: {args.dpi}")
        print(f"Cell size: {args.cell_size_mm}mm")
        print(f"Margins: {args.margin_mm}mm")
        
        if crop_rect:
            print(f"Crop: {crop_rect}")
        
        print("-" * 60)
        
        # Generate the kit
        results = generate_diamond_kit(
            image_path=args.input,
            style_name=args.style.upper(),
            output_dir=args.output,
            dpi=args.dpi,
            margin_mm=args.margin_mm,
            cell_size_mm=args.cell_size_mm,
            crop_rect=crop_rect
        )
        
        # Display results
        print("\n" + "="*60)
        print("[OK] KIT GENERATION COMPLETE")
        print("="*60)
        
        metadata = results["metadata"]
        grid_specs = results["grid_specs"]
        quality_report = results["quality_report"]
        
        print(f"Output directory: {args.output}")
        print(f"Style: {metadata.get('generation_info', {}).get('style', 'N/A')}")
        print(f"Grid size: {grid_specs.cols} x {grid_specs.rows} ({grid_specs.total_cells:,} cells)")
        print(f"Total pages: {metadata.get('print_specifications', {}).get('total_pages', 'N/A')}")
        print(f"Grid hash: {metadata.get('generation_info', {}).get('grid_hash', 'N/A')}")
        
        qa = metadata.get('quality_assessment', {})
        print(f"\nQuality Assessment:")
        print(f"  Overall Quality: {qa.get('overall_quality', 'N/A')}")
        print(f"  DeltaE Mean: {qa.get('delta_e_mean', 0):.2f}")
        print(f"  DeltaE Max: {qa.get('delta_e_max', 0):.2f}")
        print(f"  SSIM: {qa.get('ssim_score', 0):.4f}")
        scale_factor = metadata.get('scale_factor')
        if scale_factor is not None:
            print(f"  Scale Factor: {scale_factor:.3f}")
            if scale_factor < 0.1:
                print("  [WARN] Scale factor below 0.10 - consider a tighter crop or simpler image.")
            elif scale_factor < 0.5:
                print("  [WARN] Significant downscale applied to fit the 10k cell limit.")
        
        quality_warnings = metadata.get('quality_warnings', [])
        if quality_warnings:
            print(f"\nWarnings:")
            for warning in quality_warnings:
                print(f"  [WARN] {warning}")
        
        quality_risks = metadata.get('quality_risks', [])
        if quality_risks:
            print(f"\nRisks:")
            for risk in quality_risks:
                print(f"  [RISK] {risk}")
        
        gates = metadata.get('quality_gates', {})
        if gates:
            print(f"\nQuality Gates:")
            passed_txt = "PASS" if gates.get('passed') else "CHECK"
            print(f"  Status: {passed_txt}")
            for warning in gates.get('warnings', []):
                print(f"  [WARN] {warning}")
            for error in gates.get('errors', []):
                print(f"  [ERROR] {error}")
        
        print(f"\nGenerated Files:")
        for key, filename in results["outputs"].items():
            print(f"  [OK] {filename}")
        
        return True
        
    except Exception as e:
        print(f"\n[X] Error during kit generation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle utility commands
    if args.list_styles:
        list_styles()
        return
    
    if args.style_info:
        show_style_info(args.style_info)
        return
    
    # Validate arguments for kit generation
    if not validate_arguments(args):
        sys.exit(1)
    
    # Generate kit
    success = generate_kit(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
