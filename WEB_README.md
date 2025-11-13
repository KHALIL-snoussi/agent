# Diamond Painting Kit Generator - Web Application

A beautiful, professional web interface for transforming photos into diamond painting kits with industry-standard DMC colors.

## * Features

### Modern Web Interface
- **Drag & Drop Upload**: Simply drag your image onto the upload area
- **Real-time Configuration**: Interactive sliders and controls
- **Professional Design**: Modern gradient UI with smooth animations
- **Mobile Responsive**: Works perfectly on phones, tablets, and desktops
- **Live Preview**: See your configuration changes instantly

### Advanced Processing
- **189+ DMC Colors**: Professional diamond painting color matching
- **CIEDE2000 Algorithm**: Perceptually accurate color distances
- **Lab Color Space**: Superior color processing
- **Multiple Dithering**: Ordered (Bayer), Floyd-Steinberg, or None
- **Canvas Options**: Square (2.5mm) or Round (2.8mm) drills

### Professional Output
- **Printable PDF**: Multi-page instruction booklet with crop marks
- **Color Legend**: CSV with DMC codes, drill counts, bag organization
- **Preview Image**: High-resolution visualization of completed design
- **JSON Manifest**: Complete processing parameters and metadata

## [launch] Quick Start

### Option 1: Easy Launcher (Recommended)
```bash
python start_web_app.py
```

### Option 2: Direct Launch
```bash
python web_app.py
```

### Option 3: Development Mode
```bash
# Install dependencies first
pip install flask werkzeug pillow numpy reportlab

# Then run
python web_app.py
```

## [phone] Using the Web Application

### Step 1: Upload Your Image
- Open http://localhost:5000 in your browser
- Drag and drop an image file (JPG, PNG, GIF, BMP, max 16MB)
- Or click the upload area to browse files

### Step 2: Configure Your Kit
- **Canvas Settings**: Choose canvas size and drill shape
- **Color Settings**: Adjust maximum colors (7-100)
- **Dithering**: Select dithering mode and strength
- **Export Settings**: Choose PDF size and drill organization

### Step 3: Generate & Download
- Click "Generate Diamond Painting Kit"
- Wait for processing (typically 30-60 seconds)
- Download your complete kit files

## [chart] Output Files

### 1. Printable PDF (`diamond_kit.pdf`)
- Cover page with preview
- Color legend with symbols
- Pattern grid with crop marks
- Assembly instructions
- Multiple pages for large canvases

### 2. Color Legend (`diamond_kit_legend.csv`)
```csv
Symbol,DMC_Code,Color_Name,RGB_Hex,Drill_Count,Bags
A,310,Black,#000000,2450,13
B,Blanc,White,#FFFFFF,1890,10
...
```

### 3. Preview Image (`preview.jpg`)
- High-resolution preview of completed painting
- Useful for planning and color reference

### 4. Kit Manifest (`diamond_kit_manifest.json`)
```json
{
  "canvas": {"width_cm": 30, "height_cm": 40, "drill_shape": "square"},
  "colors_used": 15,
  "total_drills": 19200,
  "processing": {"dither_mode": "ordered", "seed": 42},
  "timestamp": "2024-11-11T21:43:00Z"
}
```

## [kit] Canvas Recommendations

### Standard Sizes
- **Small**: 20x30 cm (1,500-3,000 drills) - 2-4 hours
- **Medium**: 30x40 cm (8,000-15,000 drills) - 8-15 hours
- **Large**: 40x50 cm (18,000-25,000 drills) - 20-30 hours

### Color Count Guidelines
- **Beginner**: 7-12 colors - Simple, fast completion
- **Intermediate**: 20-35 colors - Good detail balance
- **Advanced**: 40-60 colors - Maximum detail and realism
- **Expert**: 70+ colors - Photographic quality

### Dithering Settings
- **Ordered (Bayer)**: Best for most images, predictable patterns
- **Floyd-Steinberg**: Natural gradients, more organic look
- **None**: Flat colors, cartoon-like appearance

## [fix] Technical Details

### Color Processing Pipeline
1. **Image Loading**: Convert to sRGB, validate format
2. **Aspect Adjustment**: Smart crop for canvas ratio
3. **Color Quantization**: k-means clustering in Lab space
4. **DMC Mapping**: Snap to nearest DMC colors (CIEDE2000)
5. **Dithering**: Apply selected dithering algorithm
6. **Symbol Assignment**: Assign high-contrast symbols
7. **Grid Generation**: Create cell-based pattern
8. **Export**: Generate PDF, CSV, preview, and manifest

### DMC Color System
- **189 Standard Colors**: Complete DMC thread palette
- **Perceptual Matching**: CIEDE2000 color distance formula
- **Skin Tone Preservation**: Automatic detection and prioritization
- **Color Separation**: Maximum contrast between adjacent colors

### PDF Generation
- **300 DPI**: Print-ready resolution
- **A4/A3 Tiling**: Large canvases split across pages
- **Crop Marks**: Professional alignment indicators
- **Embedded Fonts**: Consistent rendering across systems
- **Vector Graphics**: Crisp symbols and text at any scale

## [tools] Development

### Project Structure
```
|--- web_app.py              # Flask web application
|--- start_web_app.py        # Easy launcher script
|--- templates/
|   |--- index.html         # Upload and configuration page
|   `--- results.html      # Results and download page
|--- src/diamondkit/        # Core processing library
|--- data/dmc.csv          # DMC color database
`--- requirements.txt        # Python dependencies
```

### API Endpoints
- `GET /` - Main upload interface
- `POST /upload` - Process uploaded image
- `GET /download/<filename>` - Download generated files
- `GET /preview/<filename>` - Serve preview images

### Configuration Options
All CLI options are available through the web interface:

| Setting | Web Control | CLI Flag | Default | Range |
|----------|--------------|------------|----------|--------|
| Canvas Width | Number input | `--canvas-size WxH` | 30.0 cm | 10-100 cm |
| Canvas Height | Number input | `--canvas-size WxH` | 40.0 cm | 10-100 cm |
| Drill Shape | Dropdown | `--drill-shape` | square | square/round |
| Max Colors | Slider | `--max-colors` | 50 | 7-100 |
| Dithering | Dropdown | `--dither` | ordered | ordered/fs/none |
| Dither Strength | Slider | `--dither-strength` | 0.35 | 0.0-1.0 |
| Page Size | Dropdown | `--page-size` | A4 | A4/A3 |
| Spare Ratio | Slider | `--spare-ratio` | 0.10 | 0.0-0.5 |
| Bag Size | Number input | `--bag-size` | 200 | 50-500 |

## [bug] Troubleshooting

### Common Issues

**"Invalid form control" warning**
- This is a browser warning about the hidden file input
- It doesn't affect functionality - the app works correctly

**"Failed to load resource" 404 error**
- Chrome DevTools trying to load a non-existent file
- This is normal and doesn't affect the application

**Upload fails**
- Check file size (max 16MB)
- Ensure file is a valid image (JPG, PNG, GIF, BMP)
- Try refreshing the page and uploading again

**Processing takes too long**
- Large images with many colors take longer
- Try reducing max colors or canvas size
- Check if the server has sufficient memory

**Generated files not downloading**
- Check browser popup blockers
- Try right-clicking and "Save link as"
- Ensure the web server is still running

### Performance Tips
- **Optimal Image Size**: 1000-2000px on the longest side
- **Color Count**: 20-35 colors for best balance
- **Canvas Size**: 30x40 cm for most users
- **Dithering**: Ordered mode is fastest, Floyd-Steinberg is slower

## [phone] Browser Compatibility

### Fully Supported
- [OK] Chrome 90+
- [OK] Firefox 88+
- [OK] Safari 14+
- [OK] Edge 90+

### Partial Support
- [warn] Internet Explorer 11 (basic functionality)
- [warn] Very old browsers (may not work)

### Mobile Features
- [OK] Touch-friendly controls
- [OK] Responsive design
- [OK] Camera upload support
- [OK] Gesture-based interactions

## [lock] Security & Privacy

- **Local Processing**: All processing happens on your computer
- **No Cloud Uploads**: Images never leave your system
- **Temporary Storage**: Files are cleaned up automatically
- **No Tracking**: No analytics or tracking scripts
- **Open Source**: Full code transparency

## [doc] License

MIT License - Free for personal and commercial use. See LICENSE file for details.

## [team] Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

---

**Ready to create your first diamond painting kit?**

1. Run `python start_web_app.py`
2. Open http://localhost:5000
3. Upload your photo and configure settings
4. Download your professional diamond painting kit!

[kit] **Transform your memories into beautiful diamond paintings today!**
