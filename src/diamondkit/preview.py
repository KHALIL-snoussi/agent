"""
Preview image generation for diamond painting kits.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional
import os

from .config import Config
from .dmc import DMCColor
from .grid import CanvasGrid


class PreviewGenerator:
    """Generates preview images for diamond painting kits."""
    
    def __init__(self, config: Config):
        """Initialize preview generator with configuration."""
        self.config = config
        self.preview_size = config.export.preview_size
    
    def create_preview(self, lab_image: np.ndarray, 
                     dmc_colors: List[DMCColor],
                     canvas_grid: CanvasGrid) -> np.ndarray:
        """
        Create comprehensive preview of the diamond painting kit.
        
        Args:
            lab_image: Processed image in Lab color space
            dmc_colors: List of DMC colors used
            canvas_grid: Canvas grid with symbol assignments
            
        Returns:
            Preview image as RGB numpy array
        """
        print("Generating preview image...")
        
        # Create base mosaic preview
        mosaic = self._create_mosaic_preview(lab_image, dmc_colors, canvas_grid)
        
        # Create color legend
        legend = self._create_color_legend(dmc_colors, canvas_grid)
        
        # Combine into single preview
        preview = self._combine_previews(mosaic, legend)
        
        print(f"Preview generated: {preview.shape[1]}×{preview.shape[0]} pixels")
        return preview
    
    def _create_mosaic_preview(self, lab_image: np.ndarray,
                             dmc_colors: List[DMCColor],
                             canvas_grid: CanvasGrid) -> Image.Image:
        """Create the main mosaic preview with optional grid lines."""
        # Convert Lab to RGB
        rgb_image = self._lab_to_rgb(lab_image)
        
        # Resize to preview dimensions
        preview_w, preview_h = self.preview_size
        
        # Maintain aspect ratio
        aspect_ratio = canvas_grid.cells_w / canvas_grid.cells_h
        if aspect_ratio > preview_w / preview_h:
            # Width limited
            new_w = preview_w
            new_h = int(preview_w / aspect_ratio)
        else:
            # Height limited
            new_h = preview_h
            new_w = int(preview_h * aspect_ratio)
        
        # Resize with high quality
        pil_image = Image.fromarray(rgb_image)
        resized_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
        
        # Add grid overlay if requested
        if new_w <= 800 and new_h <= 800:  # Only for smaller previews
            resized_image = self._add_grid_overlay(resized_image, canvas_grid)
        
        # Add border
        border_size = 2
        bordered_image = Image.new(
            'RGB', 
            (new_w + 2*border_size, new_h + 2*border_size), 
            (0, 0, 0)
        )
        bordered_image.paste(resized_image, (border_size, border_size))
        
        return bordered_image
    
    def _add_grid_overlay(self, image: Image.Image, 
                         canvas_grid: CanvasGrid) -> Image.Image:
        """Add subtle grid lines to show drill positions."""
        draw = ImageDraw.Draw(image)
        w, h = image.size
        
        # Calculate grid cell size in preview
        cell_w = w / canvas_grid.cells_w
        cell_h = h / canvas_grid.cells_h
        
        # Draw vertical lines
        for x in range(canvas_grid.cells_w + 1):
            x_pos = int(x * cell_w)
            draw.line([(x_pos, 0), (x_pos, h)], fill=(200, 200, 200), width=1)
        
        # Draw horizontal lines
        for y in range(canvas_grid.cells_h + 1):
            y_pos = int(y * cell_h)
            draw.line([(0, y_pos), (w, y_pos)], fill=(200, 200, 200), width=1)
        
        # Draw bold lines every 10 cells
        line_width = 2
        
        # Vertical bold lines
        for x in range(0, canvas_grid.cells_w + 1, 10):
            x_pos = int(x * cell_w)
            draw.line([(x_pos, 0), (x_pos, h)], fill=(100, 100, 100), width=line_width)
        
        # Horizontal bold lines
        for y in range(0, canvas_grid.cells_h + 1, 10):
            y_pos = int(y * cell_h)
            draw.line([(0, y_pos), (w, y_pos)], fill=(100, 100, 100), width=line_width)
        
        return image
    
    def _create_color_legend(self, dmc_colors: List[DMCColor],
                          canvas_grid: CanvasGrid) -> Image.Image:
        """Create color legend with DMC codes and symbols."""
        legend_items = canvas_grid.get_color_legend()
        
        if not legend_items:
            return Image.new('RGB', (100, 100), (255, 255, 255))
        
        # Calculate legend dimensions
        swatch_size = 30
        padding = 10
        text_height = 20
        item_height = max(swatch_size, text_height) + padding
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 12)
            small_font = ImageFont.truetype("arial.ttf", 10)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Calculate width based on longest text
        max_text_width = 0
        for item in legend_items:
            text = f"{item['symbol']} {item['dmc_code']}"
            try:
                text_width = font.getlength(text)
            except:
                text_width = len(text) * 6  # Rough estimate
            max_text_width = max(max_text_width, text_width)
        
        legend_width = int(max_text_width + swatch_size + padding * 3)
        legend_height = len(legend_items) * item_height + padding * 2
        
        # Create legend image
        legend = Image.new('RGB', (legend_width, legend_height), (255, 255, 255))
        draw = ImageDraw.Draw(legend)
        
        # Draw title
        title = "COLOR LEGEND"
        try:
            title_width = font.getlength(title)
        except:
            title_width = len(title) * 6
        
        title_x = (legend_width - title_width) // 2
        draw.text((title_x, padding), title, fill=(0, 0, 0), font=font)
        
        # Draw color items
        y_offset = padding * 2 + text_height
        
        for item in legend_items:
            # Draw color swatch
            rgb = item['rgb']
            swatch_rect = [
                padding, 
                y_offset, 
                padding + swatch_size, 
                y_offset + swatch_size
            ]
            draw.rectangle(swatch_rect, fill=rgb, outline=(0, 0, 0))
            
            # Draw text
            text = f"{item['symbol']} {item['dmc_code']} ({item['count']})"
            text_x = padding + swatch_size + padding
            text_y = y_offset + (swatch_size - text_height) // 2
            
            draw.text(
                (text_x, text_y), 
                text, 
                fill=(0, 0, 0), 
                font=small_font
            )
            
            y_offset += item_height
        
        return legend
    
    def _combine_previews(self, mosaic: Image.Image, 
                        legend: Image.Image) -> np.ndarray:
        """Combine mosaic and legend into single preview."""
        # Calculate combined dimensions
        mosaic_w, mosaic_h = mosaic.size
        legend_w, legend_h = legend.size
        
        # Place legend to the right of mosaic
        combined_w = mosaic_w + legend_w + 20
        combined_h = max(mosaic_h, legend_h)
        
        # Create combined image
        combined = Image.new('RGB', (combined_w, combined_h), (240, 240, 240))
        
        # Paste mosaic
        mosaic_y = (combined_h - mosaic_h) // 2
        combined.paste(mosaic, (10, mosaic_y))
        
        # Paste legend
        legend_x = mosaic_w + 20
        legend_y = (combined_h - legend_h) // 2
        combined.paste(legend, (legend_x, legend_y))
        
        return np.array(combined)
    
    def _lab_to_rgb(self, lab_image: np.ndarray) -> np.ndarray:
        """Convert Lab image to RGB."""
        import cv2
        
        # Scale Lab to OpenCV ranges
        lab_copy = lab_image.copy()
        lab_copy[:, :, 0] = lab_copy[:, :, 0] * 255.0 / 100.0
        
        # Convert to RGB
        rgb_float = cv2.cvtColor(lab_copy, cv2.COLOR_LAB2RGB)
        rgb_image = (rgb_float * 255).astype(np.uint8)
        
        return rgb_image
    
    def save_preview(self, preview_image: np.ndarray, output_path: str):
        """Save preview image to file."""
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to PIL and save
        pil_image = Image.fromarray(preview_image)
        pil_image.save(output_path, quality=95)
        print(f"Preview saved: {output_path}")
    
    def create_comparison_preview(self, original_lab: np.ndarray,
                              quantized_lab: np.ndarray,
                              dithered_lab: np.ndarray,
                              canvas_grid: CanvasGrid) -> np.ndarray:
        """
        Create side-by-side comparison of processing stages.
        
        Args:
            original_lab: Original Lab image
            quantized_lab: Quantized Lab image  
            dithered_lab: Dithered Lab image
            canvas_grid: Canvas grid
            
        Returns:
            Comparison preview as RGB numpy array
        """
        # Convert all to RGB
        original_rgb = self._lab_to_rgb(original_lab)
        quantized_rgb = self._lab_to_rgb(quantized_lab)
        dithered_rgb = self._lab_to_rgb(dithered_lab)
        
        # Create grid visualization
        grid_viz = canvas_grid.create_grid_visualization()
        
        # Resize all to same dimensions
        target_size = (400, 300)  # Width, Height
        
        def resize_image(img_array):
            pil_image = Image.fromarray(img_array)
            resized = pil_image.resize(target_size, Image.LANCZOS)
            return np.array(resized)
        
        original_resized = resize_image(original_rgb)
        quantized_resized = resize_image(quantized_rgb)
        dithered_resized = resize_image(dithered_rgb)
        grid_resized = resize_image(grid_viz)
        
        # Arrange in 2x2 grid
        top_row = np.hstack([original_resized, quantized_resized])
        bottom_row = np.hstack([dithered_resized, grid_resized])
        comparison = np.vstack([top_row, bottom_row])
        
        # Add labels
        labeled_comparison = self._add_comparison_labels(comparison)
        
        return labeled_comparison
    
    def _add_comparison_labels(self, comparison: np.ndarray) -> np.ndarray:
        """Add text labels to comparison image."""
        h, w = comparison.shape[:2]
        
        # Add space for labels at top
        labeled_h = h + 60
        labeled = np.ones((labeled_h, w, 3), dtype=np.uint8) * 255
        
        # Copy image below labels
        labeled[60:, :] = comparison
        
        # Convert to PIL for text drawing
        pil_image = Image.fromarray(labeled)
        draw = ImageDraw.Draw(pil_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 14)
            title_font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
        
        # Add title
        title = "PROCESSING STAGES"
        try:
            title_width = title_font.getlength(title)
        except:
            title_width = len(title) * 8
        
        title_x = (w - title_width) // 2
        draw.text((title_x, 10), title, fill=(0, 0, 0), font=title_font)
        
        # Add labels
        half_w = w // 2
        labels = ["Original", "Quantized", "Dithered", "Grid"]
        positions = [
            (half_w // 2, 35),
            (half_w + half_w // 2, 35),
            (half_w // 2, h // 2 + 35),
            (half_w + half_w // 2, h // 2 + 35)
        ]
        
        for label, (x, y) in zip(labels, positions):
            try:
                label_width = font.getlength(label)
            except:
                label_width = len(label) * 6
            
            label_x = x - label_width // 2
            draw.text((label_x, y), label, fill=(0, 0, 0), font=font)
        
        return np.array(pil_image)
