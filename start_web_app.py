#!/usr/bin/env python3
"""
Simple launcher for the Diamond Painting Kit Generator Web Application
"""

import sys
import os
from pathlib import Path

def main():
    """Launch the web application with proper setup."""
    print("[kit] Diamond Painting Kit Generator - Web Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("[X] Error: src/ directory not found!")
        print("Please run this from the project root directory.")
        sys.exit(1)
    
    if not Path("data/dmc.csv").exists():
        print("[X] Error: data/dmc.csv not found!")
        print("Please ensure the DMC color data file is available.")
        sys.exit(1)
    
    # Check dependencies
    try:
        import flask
        print("[OK] Flask found")
    except ImportError:
        print("[package] Installing Flask...")
        os.system(f"{sys.executable} -m pip install flask werkzeug")
    
    try:
        import PIL
        print("[OK] Pillow found")
    except ImportError:
        print("[package] Installing Pillow...")
        os.system(f"{sys.executable} -m pip install pillow")
    
    try:
        import numpy
        print("[OK] NumPy found")
    except ImportError:
        print("[package] Installing NumPy...")
        os.system(f"{sys.executable} -m pip install numpy")
    
    try:
        import reportlab
        print("[OK] ReportLab found")
    except ImportError:
        print("[package] Installing ReportLab...")
        os.system(f"{sys.executable} -m pip install reportlab")
    
    print("\n[launch] Starting web application...")
    print("Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Launch the web app
    os.system(f"{sys.executable} web_app.py")

if __name__ == "__main__":
    main()
