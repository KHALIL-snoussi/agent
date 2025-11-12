"""
Commercial Diamond Painting Kit Generator - Web Interface
Provides a clean UI for uploading images and generating commercial-grade kits
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
    from diamondkit.kit_generator import generate_diamond_kit, get_available_styles
    from diamondkit.fixed_palettes import get_style_info, get_all_styles_info
    from diamondkit.print_math import get_print_math_engine, PrintSpecs
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

def generate_commercial_kit(image_path: str, style_name: str, config_overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Generate commercial diamond painting kit from image."""
    try:
        print("🎨 Starting commercial diamond painting kit generation...")
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"kit_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Set commercial-grade defaults with quality improvements
        dpi = int(config_overrides.get('dpi', 600))
        margin_mm = float(config_overrides.get('margin_mm', 12.0))
        cell_size_mm = float(config_overrides.get('cell_size_mm', 2.5))  # Smaller cells for better detail
        
        # Generate kit using commercial system
        result = generate_diamond_kit(
            image_path=image_path,
            style_name=style_name,
            output_dir=output_dir,
            dpi=dpi,
            margin_mm=margin_mm,
            cell_size_mm=cell_size_mm
        )
        
        if result and 'metadata' in result:
            metadata = result['metadata']
            grid_specs = result['grid_specs']
            quality_report = result['quality_report']
            
            # Return web-friendly results
            return {
                'success': True,
                'output_dir': output_dir,
                'style': style_name,
                'grid_size': f"{grid_specs.cols}×{grid_specs.rows}",
                'total_drills': grid_specs.total_cells,
                'total_pages': metadata['print_specifications']['total_pages'],
                'quality': quality_report['summary']['overall_quality'],
                'ssim_score': quality_report['summary']['ssim_score'],
                'delta_e_mean': quality_report['summary']['delta_e_mean'],
                'delta_e_max': quality_report['summary']['delta_e_max'],
                'preview_path': result['outputs']['original_preview'],
                'quantized_path': result['outputs']['quantized_preview'],
                'pdf_path': result['outputs']['pdf_kit'],
                'csv_path': result['outputs']['csv_inventory'],
                'json_path': result['outputs']['json_metadata'],
                'style_previews': {
                    'original': result['outputs'].get('preview_original'),
                    'vintage': result['outputs'].get('preview_vintage'),
                    'popart': result['outputs'].get('preview_popart')
                },
                'palette_info': metadata['palette_info'],
                'warnings': quality_report['quality_gates']['warnings'],
                'recommendations': quality_report['quality_gates']['auto_fixes']
            }
        
        else:
            return {'success': False, 'error': 'No result returned from kit generator'}
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.route('/')
def index():
    """Main page with upload form."""
    # Get available styles
    styles = get_available_styles()
    style_info = {}
    for style in styles:
        style_info[style] = get_style_info(style)
    
    return render_template('index.html', styles=style_info)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing."""
    try:
        print(f"DEBUG: Request method: {request.method}")
        print(f"DEBUG: Request files: {list(request.files.keys())}")
        print(f"DEBUG: Request form: {list(request.form.keys())}")
        
        if 'file' not in request.files:
            print("DEBUG: No 'file' in request.files")
            flash('No file selected')
            return redirect(url_for('index'))
        
        file = request.files['file']
        print(f"DEBUG: File object: {file}")
        print(f"DEBUG: File filename: {file.filename}")
        
        if file.filename == '':
            print("DEBUG: Empty filename")
            flash('No file selected')
            return redirect(url_for('index'))
        
        # Save uploaded file
        image_path = save_uploaded_file(file)
        if not image_path:
            flash('Invalid file type')
            return redirect(url_for('index'))
        
        # Get configuration from form
        style_name = request.form.get('style', 'ORIGINAL')
        config_overrides = {
            'dpi': int(request.form.get('dpi', 600)),
            'margin_mm': float(request.form.get('margin_mm', 12.0)),
            'cell_size_mm': float(request.form.get('cell_size_mm', 2.8))
        }
        
        # Generate commercial kit
        result = generate_commercial_kit(image_path, style_name, config_overrides)
        
        if result['success']:
            # Load color usage from JSON metadata file
            color_usage = {}
            try:
                with open(result['json_path'], 'r') as f:
                    json_data = json.load(f)
                    color_usage = json_data.get('color_usage', {})
            except Exception as e:
                print(f"Warning: Could not load color usage from JSON: {e}")
            
            # Add DMC colors to the result
            result['dmc_colors'] = [
                {
                    'code': dmc_code,
                    'name': dmc_code,  # Will be displayed as DMC code
                    'hex': '#' + str(hash(dmc_code) % 0xFFFFFF).zfill(6),  # Generate consistent color
                    'count': color_usage.get(dmc_code, 0)
                } for dmc_code in result['palette_info']['dmc_codes']
            ]
            
            # Extract filenames from paths
            result['pdf_filename'] = os.path.basename(result['pdf_path'])
            result['csv_filename'] = os.path.basename(result['csv_path'])
            result['json_filename'] = os.path.basename(result['json_path'])
            result['preview_filename'] = os.path.basename(result['preview_path'])
            result['quantized_filename'] = os.path.basename(result['quantized_path'])
            
            # Store result in session
            session_data = {
                'image_info': get_image_info(image_path),
                'kit_result': result,
                'config': config_overrides
            }
            return render_template('results.html', **session_data)
        else:
            error_msg = result.get('error', 'Unknown error occurred')
            print(f"DEBUG: Kit generation failed: {error_msg}")
            print(f"DEBUG: Result keys: {list(result.keys()) if result else 'None'}")
            flash(f"Error generating kit: {error_msg}")
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

@app.route('/api/styles')
def api_styles():
    """Get available styles via API."""
    try:
        styles = get_available_styles()
        style_info = {}
        for style in styles:
            style_info[style] = get_style_info(style)
        return jsonify(style_info)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/style/<style_name>')
def api_style_info(style_name):
    """Get specific style information via API."""
    try:
        info = get_style_info(style_name)
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    flash('File too large. Maximum size is 16MB.')
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("🎨 Commercial Diamond Painting Kit Generator - Web Interface")
    print("=" * 60)
    print("Starting web server...")
    print("Open your browser and go to: http://localhost:5000")
    print("=" * 60)
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
