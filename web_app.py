"""
Simple Web Application for Diamond Painting Kit Generator
Provides a clean UI for uploading images and generating kits
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, 'src')

try:
    from diamondkit.config import Config
    from diamondkit.image_io import ImageLoader
    from diamondkit.dmc import get_dmc_palette
    from diamondkit.quantize import ColorQuantizer
    from diamondkit.dither import DitherEngine
    from diamondkit.grid import CanvasGrid
    from diamondkit.preview import PreviewGenerator
    from diamondkit.export import ExportManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install dependencies: pip install -r requirements.txt")
    sys.exit(1)

# Web framework - using Flask for simplicity
try:
    from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for, session
    from werkzeug.utils import secure_filename
    from werkzeug.exceptions import RequestEntityTooLarge
except ImportError:
    print("Installing Flask for web interface...")
    os.system("pip install flask werkzeug")
    from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for, session
    from werkzeug.utils import secure_filename
    from werkzeug.exceptions import RequestEntityTooLarge

# Image processing
from PIL import Image
import io
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'diamond-painting-kit-generator-2024'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(file) -> str:
    """Save uploaded file and return path."""
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return filepath
    return None

def get_image_info(image_path: str) -> Dict[str, Any]:
    """Get basic image information."""
    try:
        with Image.open(image_path) as img:
            return {
                'width': img.width,
                'height': img.height,
                'format': img.format,
                'mode': img.mode,
                'size_bytes': os.path.getsize(image_path),
                'aspect_ratio': img.width / img.height
            }
    except Exception as e:
        return {'error': str(e)}

def generate_kit_from_image(image_path: str, config_overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Generate diamond painting kit from image."""
    try:
        # Create configuration
        config = Config()
        
        # Apply overrides
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif hasattr(config.canvas, key):
                setattr(config.canvas, key, value)
            elif hasattr(config.palette, key):
                setattr(config.palette, key, value)
            elif hasattr(config.dither, key):
                setattr(config.dither, key, value)
            elif hasattr(config.export, key):
                setattr(config.export, key, value)
        
        # Set input
        config.input = image_path
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.output_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"kit_{timestamp}")
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize components
        dmc_palette = get_dmc_palette("data/dmc.csv")
        image_loader = ImageLoader(config)
        quantizer = ColorQuantizer(config, dmc_palette)
        dither_engine = DitherEngine(config)
        
        print("🎨 Starting diamond painting kit generation...")
        
        # Load and process image
        print("📸 Loading and preprocessing image...")
        image_lab, metadata = image_loader.load_image(image_path)
        
        # Quantize colors
        print("🎨 Optimizing color palette...")
        quantized_lab, dmc_colors = quantizer.quantize_image(image_lab)
        
        # Apply dithering
        print("✨ Applying dithering for smooth gradients...")
        dithered_lab = dither_engine.apply_dithering(image_lab, quantized_lab)
        
        # Create grid
        print("📐 Creating diamond placement grid...")
        canvas_grid = CanvasGrid(config, dmc_colors)
        grid_data = canvas_grid.create_grid(dithered_lab)
        
        # Generate preview
        print("👁️ Generating preview image...")
        preview_generator = PreviewGenerator(config)
        preview_image = preview_generator.create_preview(dithered_lab, dmc_colors, canvas_grid)
        
        # Export files
        print("📄 Creating printable PDF and materials...")
        export_manager = ExportManager(config)
        export_manager.export_complete_kit(canvas_grid, preview_image, metadata)
        
        print("✅ Diamond painting kit generation complete!")
        
        # Return results
        return {
            'success': True,
            'output_dir': config.output_dir,
            'colors_used': len(dmc_colors),
            'grid_size': f"{canvas_grid.cells_w}×{canvas_grid.cells_h}",
            'total_drills': canvas_grid.cells_w * canvas_grid.cells_h,
            'preview_path': os.path.join(config.output_dir, 'preview.jpg'),
            'pdf_path': os.path.join(config.output_dir, 'diamond_kit.pdf'),
            'csv_path': os.path.join(config.output_dir, 'diamond_kit_legend.csv'),
            'json_path': os.path.join(config.output_dir, 'diamond_kit_manifest.json'),
            'dmc_colors': [
                {
                    'code': color.dmc_code,
                    'name': color.name,
                    'rgb': color.rgb,
                    'hex': color.hex
                } for color in dmc_colors
            ]
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.route('/')
def index():
    """Main page with upload form."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing."""
    try:
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(url_for('index'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(url_for('index'))
        
        # Save uploaded file
        image_path = save_uploaded_file(file)
        if not image_path:
            flash('Invalid file type')
            return redirect(url_for('index'))
        
        # Get configuration from form
        config_overrides = {
            'canvas.width_cm': float(request.form.get('canvas_width', 30.0)),
            'canvas.height_cm': float(request.form.get('canvas_height', 40.0)),
            'canvas.drill_shape': request.form.get('drill_shape', 'square'),
            'palette.max_colors': int(request.form.get('max_colors', 50)),
            'dither.mode': request.form.get('dither_mode', 'ordered'),
            'dither.strength': float(request.form.get('dither_strength', 0.35)),
            'export.page': request.form.get('page_size', 'A4'),
            'export.spare_ratio': float(request.form.get('spare_ratio', 0.10)),
            'export.bag_size': int(request.form.get('bag_size', 200))
        }
        
        # Generate kit
        result = generate_kit_from_image(image_path, config_overrides)
        
        if result['success']:
            # Store result in session
            session_data = {
                'image_info': get_image_info(image_path),
                'kit_result': result,
                'config': config_overrides
            }
            return render_template('results.html', **session_data)
        else:
            flash(f"Error generating kit: {result['error']}")
            return redirect(url_for('index'))
    
    except RequestEntityTooLarge:
        flash('File too large. Maximum size is 16MB.')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f"Error processing file: {str(e)}")
        return redirect(url_for('index'))

@app.route('/download/<path:filename>')
def download_file(filename):
    """Download generated files."""
    try:
        # Find the file in upload directory
        for root, dirs, files in os.walk(app.config['UPLOAD_FOLDER']):
            if filename in files:
                return send_file(os.path.join(root, filename), as_attachment=True)
        flash('File not found')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f"Error downloading file: {str(e)}")
        return redirect(url_for('index'))

@app.route('/preview/<path:filename>')
def serve_preview(filename):
    """Serve preview image."""
    try:
        for root, dirs, files in os.walk(app.config['UPLOAD_FOLDER']):
            if filename in files:
                return send_file(os.path.join(root, filename))
        return "File not found", 404
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/api/image_info')
def api_image_info():
    """Get image information via API."""
    try:
        # This would need the actual image data from upload
        return jsonify({'error': 'Not implemented'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    flash('File too large. Maximum size is 16MB.')
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("🎨 Diamond Painting Kit Generator - Web Interface")
    print("=" * 50)
    print("Starting web server...")
    print("Open your browser and go to: http://localhost:5000")
    print("=" * 50)
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
