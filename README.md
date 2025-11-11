# Paint-by-Numbers / Diamond-Painting Generator

A professional-grade tool that converts any photo into a print-ready paint-by-numbers or diamond-painting kit. Transform your memories into beautiful craft projects with our end-to-end processing pipeline.

## 🎨 Features

- **Smart Image Processing**: Automatic cropping, color quantization, and dithering
- **Professional Color Palette**: 7-color palette with DMC-equivalent codes
- **Print-Ready PDF**: Multi-page kits with legends, instructions, and tiled patterns
- **Customizable**: YAML configuration for canvas size, colors, and output settings
- **High Quality**: 300 DPI output with proper symbol rendering
- **Deterministic**: Same input + config = identical output every time

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd paint-by-numbers-generator

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Generate a kit from your image
python main.py generate input_photo.jpg output_kit.pdf

# Create a demo kit to test
python main.py demo

# Show current color palette
python main.py palette

# Validate an image before processing
python main.py validate input_photo.jpg
```

## 📋 Commands

### `generate` - Create a paint-by-numbers kit

```bash
python main.py generate <input_image> <output_pdf> [options]
```

**Options:**
- `--config, -c`: Configuration file path (default: config.yaml)
- `--save-intermediate, -s`: Save intermediate processing images
- `--palette-info, -p`: Show palette information

**Example:**
```bash
python main.py generate photo.jpg kit.pdf --save-intermediate --config my_config.yaml
```

### `demo` - Create a demonstration kit

```bash
python main.py demo
```

Creates a sample image and generates a complete kit to showcase the capabilities.

### `palette` - Display color palette

```bash
python main.py palette [--config config.yaml]
```

Shows the current color palette with RGB values and DMC codes.

### `validate` - Check image compatibility

```bash
python main.py validate <input_image> [--config config.yaml]
```

Validates that an image meets the requirements for processing.

### `init-config` - Create default configuration

```bash
python main.py init-config [--output config.yaml]
```

Creates a default configuration file with all settings.

## ⚙️ Configuration

The generator uses a YAML configuration file (`config.yaml`) to customize all aspects of the kit generation:

### Canvas Settings
```yaml
canvas:
  width_cm: 30          # Canvas width in cm
  height_cm: 40         # Canvas height in cm
  drill_size_mm: 2.5    # Size of each drill/dot
  aspect_ratio: 0.75    # 3:4 aspect ratio
```

### Color Palette
```yaml
palette:
  colors:
    - name: "Black"
      rgb: [0, 0, 0]
      dmc_code: "310"
    # ... more colors
```

### Processing Settings
```yaml
processing:
  dpi: 300                     # Output DPI
  color_space: "Lab"           # Lab or RGB
  quantization_method: "kmeans" # kmeans or median_cut
  dithering: true              # Apply error diffusion
  seed: 42                     # Random seed
```

### PDF Settings
```yaml
pdf:
  page_size: "A4"      # A4 or A3
  tiling: true         # Split large patterns
  overlap_mm: 5        # Overlap between tiles
  crop_marks: true     # Add alignment marks
  margins_mm: 10       # Page margins
```

## 🎯 Technical Details

### Image Processing Pipeline

1. **Preprocessing**: Auto-orientation, RGB conversion, smart cropping to 3:4 aspect ratio
2. **Color Quantization**: Lab color space conversion with k-means clustering to fixed palette
3. **Symbol Assignment**: High-contrast symbols mapped to each color
4. **Count Calculation**: Cell counting with configurable spare percentage

### Canvas Mathematics

For a 30×40 cm canvas with 2.5mm drills:
- **Width**: 300mm ÷ 2.5mm = 120 drills
- **Height**: 400mm ÷ 2.5mm = 160 drills
- **Total**: 19,200 drills per kit

### Color System

- **7-color palette**: Optimized for beginner-friendly projects
- **DMC compatibility**: Standard embroidery thread codes
- **Perceptual quantization**: Lab color space for better visual results
- **High contrast**: Symbols designed for readability

## 📊 Output Structure

Generated PDFs include:

1. **Cover Page**: Preview image and kit information
2. **Color Legend**: Swatches, symbols, names, DMC codes, and quantities
3. **Instructions**: Step-by-step assembly guide
4. **Pattern Pages**: Symbol grid with tiling for large patterns

## 🎨 Customization Guide

### Changing the Color Palette

Edit the `palette.colors` section in `config.yaml`:

```yaml
palette:
  colors:
    - name: "Custom Blue"
      rgb: [30, 144, 255]  # RGB values (0-255)
      dmc_code: "800"       # DMC thread code
    # Add up to any number of colors
```

### Adjusting Canvas Size

```yaml
canvas:
  width_cm: 25   # Smaller canvas
  height_cm: 35
  drill_size_mm: 2.0  # Smaller drills for more detail
```

### PDF Customization

```yaml
pdf:
  page_size: "A3"        # Larger pages
  tiling: false          # Single page pattern
  margins_mm: 15         # Wider margins
```

## 🔬 Research Insights

### Product Standards Analysis

**Market Research Findings:**
- **Standard Sizes**: 30×40 cm most popular for home projects
- **Drill Size**: 2.5mm standard for diamond painting
- **Color Count**: 7-12 colors optimal for beginners vs. 20+ for advanced
- **Completion Time**: 8-12 hours for standard 30×40 cm kit

**Competitor Best Practices:**
- **QBRIX**: Excellent symbol clarity and PDF organization
- **Diamond Art Club**: Professional color legends and DMC integration
- **Paint Plot**: Clear instructions and progress tracking

### Technical Decisions

**Color Quantization**:
- Lab color space chosen over RGB for perceptual accuracy
- K-means clustering provides consistent results across images
- Floyd-Steinberg dithering optional for gradient preservation

**PDF Generation**:
- 300 DPI standard for print quality
- A4 tiling with 5mm overlap for home printing
- Embedded fonts ensure consistent rendering

## 🛠️ Development

### Project Structure

```
paint-by-numbers-generator/
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── image_processor.py # Image processing pipeline
│   ├── pdf_generator.py   # PDF generation
│   ├── paint_generator.py # Main coordinator
│   └── cli.py            # Command-line interface
├── config.yaml           # Default configuration
├── main.py              # Entry point
├── requirements.txt      # Dependencies
└── README.md           # This file
```

### Dependencies

- **Pillow**: Image processing and manipulation
- **NumPy**: Numerical operations
- **OpenCV**: Image resizing and processing
- **scikit-image**: Advanced image processing
- **ReportLab**: PDF generation
- **Click**: Command-line interface
- **PyYAML**: Configuration file parsing

## 📝 Examples

### Basic Kit Generation

```bash
# Simple kit from photo
python main.py generate vacation.jpg vacation_kit.pdf

# With intermediate images for debugging
python main.py generate portrait.jpg portrait_kit.pdf --save-intermediate

# Custom configuration
python main.py generate landscape.jpg landscape_kit.pdf --config pro_config.yaml
```

### Batch Processing

```bash
# Process multiple images (bash loop)
for img in *.jpg; do
    python main.py generate "$img" "${img%.jpg}_kit.pdf"
done
```

## 🐛 Troubleshooting

### Common Issues

**"Image too small" error:**
- Minimum input size: 100×100 pixels
- Recommended size: 1200×1600 pixels or larger

**Low color diversity:**
- Try images with varied colors
- Increase drill size in config for more detail
- Disable dithering for cleaner blocks

**PDF quality issues:**
- Ensure 300 DPI setting in config
- Use vector-friendly symbols
- Check printer settings

### Performance Tips

- **Large Images**: Resize before processing for faster results
- **Memory Usage**: Process one image at a time
- **PDF Size**: Use A4 tiling for large patterns

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For questions, issues, or feature requests:
1. Check the troubleshooting section
2. Review the configuration examples
3. Create an issue in the repository

---

**Transform your photos into beautiful craft projects with professional-grade paint-by-numbers kits!** 🎨✨
