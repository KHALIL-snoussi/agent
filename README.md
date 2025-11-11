# Diamond Painting Kit Generator

A professional-grade tool that transforms any image into a complete diamond painting kit with DMC color mapping, symbol grids, and print-ready PDF instructions.

## 🎯 Features

- **Professional DMC Integration**: Uses official DMC color palette with 447+ colors
- **Advanced Color Processing**: Lab color space quantization with CIEDE2000 distance matching
- **Smart Dithering**: Ordered (Bayer) and Floyd-Steinberg dithering options
- **Print-Ready PDF**: Multi-page instruction booklets with A4/A3 tiling, crop marks, and assembly guides
- **Flexible Output**: CSV legends, JSON manifests, and preview images
- **Configurable**: Square/round drills, customizable canvas sizes, spare drill calculations

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/KHALIL-snoussi/agent.git
cd agent

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Generate a diamond painting kit from an image
python -m diamondkit.cli input.jpg output_folder/

# Custom canvas size and colors
python -m diamondkit.cli input.jpg output/ --canvas-size 40x50 --max-colors 60

# Use round drills and Floyd-Steinberg dithering
python -m diamondkit.cli input.jpg output/ --drill-shape round --dither fs
```

### Demo

```bash
# Run demo to test system
python demo_diamondkit.py

# Full demo with sample image (requires pixel.jpg)
python demo_diamondkit.py --full
```

## 📋 Command Line Options

### Input/Output
- `input`: Input image file (JPG/PNG)
- `output`: Output directory for generated kit

### Canvas Settings
- `--canvas-size`: Canvas size in cm (default: 30x40)
- `--drill-shape`: Drill shape - square/round (default: square)
- `--drill-size-mm`: Drill size in mm (default: 2.5 for square, 2.8 for round)

### Color Settings
- `--max-colors`: Maximum DMC colors (default: 50)
- `--preserve-skin-tones`: Preserve skin tones in quantization (default: True)
- `--no-preserve-skin-tones`: Disable skin tone preservation

### Dithering Options
- `--dither`: Dithering mode - none/ordered/fs (default: ordered)
- `--dither-strength`: Dithering strength 0-1 for ordered mode (default: 0.35)

### Export Settings
- `--page-size`: PDF page size - A4/A3 (default: A4)
- `--spare-ratio`: Spare drill ratio 0-1 (default: 0.10)
- `--bag-size`: Drills per bag (default: 200)

### Utility Commands
- `--validate-only`: Only validate input image
- `--info`: Show detailed image information
- `--verbose`: Enable verbose output
- `--config`: Custom YAML configuration file

## 📊 Output Files

Each generated kit includes:

1. **PDF Instruction Booklet** (`diamond_kit_[ID].pdf`)
   - Title page with preview and specifications
   - Color legend with drill counts and bag requirements
   - Step-by-step assembly instructions
   - Multi-page pattern with symbols and crop marks

2. **CSV Legend** (`diamond_kit_[ID]_legend.csv`)
   - Symbol mappings and DMC codes
   - Drill counts and bag calculations
   - Color names and hex values

3. **JSON Manifest** (`diamond_kit_[ID]_manifest.json`)
   - Complete kit parameters and metadata
   - Processing settings and quality metrics
   - Color palette and grid information

4. **Preview Image** (`diamond_kit_[ID]_preview.jpg`)
   - Visual representation of completed design
   - Color legend and grid overlay

## 🎨 Color Processing Pipeline

### 1. Image Preprocessing
- Auto-orientation based on EXIF data
- Smart cropping to match canvas aspect ratio
- Contrast and exposure normalization
- Conversion to Lab color space for perceptual accuracy

### 2. Color Quantization
- **K-means clustering** with DMC palette constraints
- **CIEDE2000 color distance** for accurate color matching
- **Skin tone preservation** for portraits and people
- Configurable color limits (7-100 colors typical)

### 3. Dithering
- **Ordered dithering** (Bayer 8×8) for consistent patterns
- **Floyd-Steinberg dithering** for natural gradients
- Adjustable dithering strength for artistic control

### 4. Symbol Assignment
- High-contrast symbol set for clarity
- Automatic symbol allocation based on color usage
- Optimized to minimize symbol confusion

## ⚙️ Configuration

### YAML Configuration

Create `config.yaml` for custom settings:

```yaml
input: "your_image.jpg"
output_dir: "output"

canvas:
  width_cm: 30.0
  height_cm: 40.0
  drill_shape: "square"  # square/round
  drill_size_mm: 2.5

palette:
  dmc_file: "data/dmc.csv"
  max_colors: 50
  preserve_skin_tones: true

dither:
  mode: "ordered"  # none/ordered/fs
  strength: 0.35

processing:
  seed: 42
  color_space: "Lab"
  quantization_method: "kmeans"

export:
  page: "A4"  # A4/A3
  pdf_dpi: 300
  margin_mm: 15
  overlap_mm: 5
  spare_ratio: 0.10
  bag_size: 200
  preview_size: [800, 600]
```

### Canvas Sizes

Common canvas dimensions:
- **Small**: 20×25 cm (1600×2000 drills)
- **Medium**: 30×40 cm (2400×3200 drills)
- **Large**: 40×50 cm (3200×4000 drills)
- **Extra Large**: 50×70 cm (4000×5600 drills)

### Drill Specifications
- **Square drills**: 2.5mm × 2.5mm
- **Round drills**: 2.8mm diameter
- **Drill density**: ~1600 drills per 10×10 cm area

## 🔧 Advanced Features

### DMC Color Integration

- **447+ official DMC colors** with Lab coordinates
- **Automatic color matching** using CIEDE2000 distance
- **Skin tone detection** for portrait accuracy
- **Color popularity weighting** for better results

### Quality Assessment

- **Automatic difficulty rating** (Beginner to Expert)
- **Complexity scoring** based on size and color count
- **Recommended experience levels** for crafters

### Export Flexibility

- **Multi-page PDF tiling** with precise alignment
- **Crop marks and registration** for professional printing
- **Coordinate labels** for easy navigation
- **Embedded fonts** for consistent rendering

## 📈 Performance

Typical processing times:
- **Small image** (800×600, 20 colors): ~2-5 seconds
- **Medium image** (1200×900, 40 colors): ~5-10 seconds
- **Large image** (2000×1500, 60 colors): ~10-20 seconds

Memory usage scales with image resolution and color count.

## 🛠️ Development

### Project Structure

```
src/diamondkit/
├── __init__.py          # Package initialization
├── config.py            # Configuration management
├── dmc.py              # DMC color palette handling
├── image_io.py          # Image loading and preprocessing
├── quantize.py          # Color quantization algorithms
├── dither.py           # Dithering engine
├── grid.py             # Canvas grid generation
├── preview.py          # Preview image generation
├── pdf.py              # PDF generation
├── export.py           # Export management
└── cli.py              # Command-line interface
```

### Core Algorithms

1. **CIEDE2000 Color Distance**
   - Perceptually accurate color difference calculation
   - Optimized for Lab color space

2. **Constrained K-means**
   - Clustering with DMC palette constraints
   - Intelligent initialization with skin tone preservation

3. **Ordered Dithering**
   - Bayer matrix implementation
   - Adjustable strength for artistic control

4. **Grid Symbol Assignment**
   - High-contrast symbol generation
   - Minimization of adjacent symbol confusion

### Testing

```bash
# Run unit tests (when implemented)
python -m pytest tests/

# Run demo tests
python demo_diamondkit.py --full
```

## 📚 Research & Best Practices

### Industry Standards

This tool incorporates research from leading diamond painting companies:

- **QBRIX**: Advanced color quantization methods
- **Paint Plot**: Professional PDF layout standards
- **Diamond Art Club**: Industry standard drill specifications

### Color Science

- **Lab color space** for perceptual uniformity
- **CIEDE2000** for accurate color difference measurement
- **Skin tone preservation** using facial color analysis

### Print Standards

- **300 DPI** for crisp symbol rendering
- **A4/A3 tiling** with proper overlap
- **Crop marks** for professional alignment
- **Embedded fonts** for consistent typography

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source. Please see LICENSE file for details.

## 🆘 Support

- **Issues**: Report bugs and feature requests on GitHub
- **Questions**: Check the demo script and documentation
- **Troubleshooting**: Run `python demo_diamondkit.py` to verify your setup

## 🎯 Future Enhancements

- [ ] Paint-by-numbers mode (traditional painting)
- [ ] Advanced symbol sets (icons, shapes)
- [ ] Batch processing for multiple images
- [ ] Web interface for easier use
- [ ] Mobile app companion
- [ ] Integration with printing services

---

**Transform your photos into stunning diamond painting art!** 🎨✨
