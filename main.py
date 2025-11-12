#!/usr/bin/env python3
"""
Main entry point for the QBRIX Diamond Painting Kit Generator.
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from diamondkit.cli import main as cli_main

if __name__ == '__main__':
    cli_main()
