#!/usr/bin/env python3
"""
QBRIX Diamond Painting Kit Demo
Demonstrates the complete fixed 7-color DMC palette workflow with quality gates.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, 'src')

from diamondkit.kit_generator import generate_diamond_kit, get_available_styles, get_style_info


def demo_style_info():
    """Display information about available QBRIX styles."""
    print("=" * 60)
    print("QBRIX DIAMOND PAINTING STYLES")
    print("=" * 60)
    
    for style in get_available_styles():
        info = get_style_info(style)
        print(f"\n{style.upper()}:")
        print(f"  Description: {info['description']}")
        print(f"  Rationale: {info['rationale']}")
        print(f"  DMC Palette: {', '.join(info['dmc_codes'])}")
        print("  Colors:")
        for i, color in enumerate(info['colors'], 1):
            print(f"    {i}. {color['dmc_code']}: {color['name']} ({color['hex']})")
    
    print("\n" + "=" * 60)


def demo_generate_kit(image_path: str, style: str, output_dir: str, 
                   dpi: int = 600, cell_size: float = 2.8, margin: float = 12.0):
    """
    Demonstrate complete QBRIX kit generation with fixed 7-color palettes.
    
    Args:
        image_path: Input image path
        style: Style name (ORIGINAL, VINTAGE, POPART)
        output_dir: Output directory
        dpi: Print DPI
        cell_size: Cell size in mm
        margin: Paper margins in mm
    """
    print("=" * 60)
    print("QBRIX KIT GENERATION DEMO")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Style: {style}")
    print(f"Output: {output_dir}")
    print(f"Print Settings: {dpi} DPI, {cell_size}mm cells, {margin}mm margins")
    print()
    
    # Generate the kit
    try:
        results = generate_diamond_kit(
            image_path=image_path,
            style_name=style.upper(),
            output_dir=output_dir,
            dpi=dpi,
            margin_mm=margin,
            cell_size_mm=cell_size
        )
        
        # Display results summary
        metadata = results["metadata"]
        grid_specs = results["grid_specs"]
        quality_report = results["quality_report"]
        
        print("\n" + "=" * 60)
        print("GENERATION RESULTS")
        print("=" * 60)
        
        print(f"[OK] Kit generated successfully!")
        print(f"[OK] Grid size: {grid_specs.cols} x {grid_specs.rows} ({grid_specs.total_cells:,} cells)")
        print(f"[OK] Total pages: {metadata['print_specifications']['total_pages']}")
        print(f"[OK] Style: {metadata['generation_info']['style']}")
        print(f"[OK] Grid hash: {metadata['generation_info']['grid_hash']}")
        
        # Quality metrics
        quality_assessment = metadata['quality_assessment']
        print(f"\nQuality Assessment:")
        print(f"  Overall Quality: {quality_assessment['overall_quality']}")
        print(f"  DeltaE2000 Mean: {metadata['quality_assessment'].get('delta_e_mean', 0):.2f}")
        print(f"  DeltaE2000 Max: {metadata['quality_assessment'].get('delta_e_max', 0):.2f}")
        print(f"  SSIM: {metadata['ssim']:.4f}")
        print(f"  Scale Factor: {metadata['scale_factor']:.3f}")
        
        # Warnings and risks
        if quality_assessment['warnings']:
            print(f"\nWarnings:")
            for warning in quality_assessment['warnings']:
                print(f"  [WARN] {warning}")
        
        if quality_assessment['risks']:
            print(f"\nRisks:")
            for risk in quality_assessment['risks']:
                print(f"  [WARN] {risk}")
        
        # Quality gates
        if 'quality_gates' in metadata:
            quality_gates = metadata['quality_gates']
            print(f"\nQuality Gates:")
            print(f"  Passed: {'[OK]' if quality_gates['passed'] else '[X]'}")
            
            if quality_gates['warnings']:
                for warning in quality_gates['warnings']:
                    print(f"  [WARN] {warning}")
            
            if quality_gates['errors']:
                for error in quality_gates['errors']:
                    print(f"  [X] {error}")
        
        # Generated files
        print(f"\nGenerated Files:")
        for key, filename in metadata['output_files'].items():
            file_path = os.path.join(output_dir, filename)
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  [OK] {filename} ({size_mb:.1f} MB)")
            else:
                print(f"  [X] {filename} (missing)")
        
        # Color usage
        print(f"\nColor Usage (7-color fixed palette):")
        color_distribution = metadata.get('quality_assessment', {}).get('color_distribution', {})
        total_used = sum(1 for pct in color_distribution.values() if pct > 0)
        print(f"  Colors used: {total_used}/7")
        
        rare_colors = metadata.get('quality_assessment', {}).get('rare_colors', [])
        if rare_colors:
            print(f"  Rare colors (< 2%): {len(rare_colors)}")
            for rare_color in rare_colors[:3]:  # Show first 3
                print(f"    - {rare_color}")
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETE")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print("Files generated:")
        print("  - diamond_painting_kit.pdf - QBRIX instruction booklet")
        print("  - inventory.csv - Drill inventory with bag quantities")
        print("  - kit_metadata.json - Complete kit metadata")
        print("  - original_preview.jpg - Input image preview")
        print("  - quantized_preview.jpg - Palette-mapped preview")
        print("  - original_style_preview.jpg - Original style preview")
        print("  - vintage_style_preview.jpg - Vintage style preview")
        print("  - popart_style_preview.jpg - Popart style preview")
        
        return results
        
    except Exception as e:
        print(f"\n[X] Kit generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def demo_all_styles(image_path: str, base_output_dir: str):
    """Generate kits for all three styles to demonstrate palette differences."""
    print("=" * 60)
    print("QBRIX ALL STYLES DEMO")
    print("=" * 60)
    print(f"Generating kits for all styles using: {image_path}")
    print(f"Base output directory: {base_output_dir}")
    print()
    
    styles = ['ORIGINAL', 'VINTAGE', 'POPART']
    results = {}
    
    for style in styles:
        style_dir = os.path.join(base_output_dir, f"demo_{style.lower()}")
        print(f"\n{'-' * 40}")
        print(f"Generating {style} style kit...")
        print(f"Output: {style_dir}")
        print('-' * 40)
        
        try:
            result = demo_generate_kit(
                image_path=image_path,
                style=style,
                output_dir=style_dir,
                dpi=300,  # Use 300 DPI for demo (faster)
                cell_size=2.8,
                margin=12.0
            )
            results[style] = result
            
            if result:
                # Brief summary for this style
                metadata = result["metadata"]
                grid_specs = result["grid_specs"]
                
                print(f"\n{style} Summary:")
                print(f"  Grid: {grid_specs.cols}x{grid_specs.rows} ({grid_specs.total_cells:,} cells)")
                print(f"  DeltaE Mean: {metadata['quality_assessment'].get('delta_e_mean', 0):.2f}")
                print(f"  SSIM: {metadata['ssim']:.3f}")
                print(f"  Quality: {metadata['quality_assessment']['overall_quality']}")
                
        except Exception as e:
            print(f"[X] {style} generation failed: {e}")
    
    # Final comparison
    print("\n" + "=" * 60)
    print("STYLE COMPARISON")
    print("=" * 60)
    
    if results:
        print(f"{'Style':<10} {'Cells':<8} {'DeltaE Mean':<9} {'SSIM':<6} {'Quality':<12}")
        print("-" * 50)
        
        for style, result in results.items():
            if result:
                metadata = result["metadata"]
                grid_specs = result["grid_specs"]
                
                print(f"{style:<10} {grid_specs.total_cells:<8,} "
                      f"{metadata['deltaE_stats']['mean']:<9.2f} "
                      f"{metadata['ssim']:<6.3f} "
                      f"{metadata['quality_assessment']['overall_quality']:<12}")
    
    return results


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description="QBRIX Diamond Painting Kit Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show available styles
  python demo_qbrix_kit.py --styles
  
  # Generate kit for specific style
  python demo_qbrix_kit.py pixel.jpg --style original --output demo_output
  
  # Generate all styles
  python demo_qbrix_kit.py pixel.jpg --all-styles --output demo_all
  
  # Custom print settings
  python demo_qbrix_kit.py pixel.jpg --style vintage --output demo_vintage --dpi 600 --cell-size 2.5
        """
    )
    
    parser.add_argument(
        "image",
        nargs="?",
        help="Input image file (JPG/PNG)"
    )
    
    parser.add_argument(
        "--style", "-s",
        choices=["original", "vintage", "popart"],
        help="Style to generate"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output directory"
    )
    
    parser.add_argument(
        "--styles", "--list-styles",
        action="store_true",
        help="Show available styles and their palettes"
    )
    
    parser.add_argument(
        "--all-styles",
        action="store_true",
        help="Generate kits for all styles"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Print DPI (default: 600)"
    )
    
    parser.add_argument(
        "--cell-size",
        type=float,
        default=2.8,
        help="Cell size in mm (default: 2.8)"
    )
    
    parser.add_argument(
        "--margin",
        type=float,
        default=12.0,
        help="Paper margins in mm (default: 12.0)"
    )
    
    args = parser.parse_args()
    
    # Handle utility commands
    if args.styles:
        demo_style_info()
        return
    
    # Validate required arguments
    if not args.image:
        print("Error: Input image required")
        parser.print_help()
        return
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    # Set default output if not provided
    if not args.output:
        base_name = Path(args.image).stem
        args.output = f"demo_{base_name}"
    
    # Generate demo
    if args.all_styles:
        demo_all_styles(args.image, args.output)
    elif args.style:
        demo_generate_kit(
            args.image, args.style, args.output, 
            args.dpi, args.cell_size, args.margin
        )
    else:
        print("Error: Must specify --style or --all-styles")
        parser.print_help()


if __name__ == "__main__":
    main()
