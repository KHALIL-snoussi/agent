"""
PDF generation module for creating print-ready paint-by-numbers kits.
Handles multi-page PDF generation with tiling, legends, and instructions.
"""

from reportlab.lib.pagesizes import A4, A3
from reportlab.lib.units import mm, inch
from reportlab.lib.colors import Color, black, white
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.platypus import Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import numpy as np
from PIL import Image
import io
from typing import List, Tuple, Dict
import os

from .config import Config


class PDFGenerator:
    """Generates print-ready PDF kits."""
    
    def __init__(self, config: Config):
        self.config = config
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        # Page dimensions
        if config.pdf.page_size == "A3":
            self.page_size = A3
        else:
            self.page_size = A4
        
        self.page_width, self.page_height = self.page_size
        self.margin = config.pdf.margins_mm * mm
        self.content_width = self.page_width - 2 * self.margin
        self.content_height = self.page_height - 2 * self.margin
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=20,
            alignment=1  # Center
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20
        ))
    
    def generate_pdf_kit(self, output_path: str, quantized_image: np.ndarray, 
                        symbol_grid: np.ndarray, color_counts: List[int],
                        original_image_path: str = None):
        """
        Generate complete PDF kit with all sections.
        """
        doc = SimpleDocTemplate(
            output_path,
            pagesize=self.page_size,
            leftMargin=self.margin,
            rightMargin=self.margin,
            topMargin=self.margin,
            bottomMargin=self.margin
        )
        
        story = []
        
        # 1. Cover page with preview
        story.extend(self._create_cover_page(quantized_image, original_image_path))
        
        # 2. Color legend
        if self.config.output.include_legend:
            story.extend(self._create_color_legend(color_counts))
        
        # 3. Instructions
        if self.config.output.include_instructions:
            story.extend(self._create_instructions())
        
        # 4. Pattern pages (with tiling if needed)
        story.extend(self._create_pattern_pages(symbol_grid))
        
        # Build PDF
        doc.build(story)
    
    def _create_cover_page(self, quantized_image: np.ndarray, original_image_path: str = None) -> List:
        """Create cover page with preview image."""
        story = []
        
        # Title
        story.append(Paragraph("Paint-by-Numbers Kit", self.styles['CustomTitle']))
        
        # Canvas info
        canvas_info = f"Canvas Size: {self.config.canvas.width_cm}×{self.config.canvas.height_cm} cm"
        story.append(Paragraph(canvas_info, self.styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # Preview image
        preview_size = self.config.output.preview_size
        preview_img = self._numpy_to_pil(quantized_image, preview_size)
        preview_buffer = io.BytesIO()
        preview_img.save(preview_buffer, format='PNG')
        preview_buffer.seek(0)
        
        rl_image = RLImage(preview_buffer, width=preview_size[0], height=preview_size[1])
        story.append(rl_image)
        story.append(Spacer(1, 20))
        
        # Kit information
        info_text = f"""
        <b>Kit Details:</b><br/>
        • Color Count: {len(self.config.palette)}<br/>
        • Drill Size: {self.config.canvas.drill_size_mm} mm<br/>
        • Total Pattern Size: {quantized_image.shape[1]}×{quantized_image.shape[0]} drills<br/>
        • Difficulty: {'Beginner' if len(self.config.palette) <= 7 else 'Intermediate'}
        """
        story.append(Paragraph(info_text, self.styles['Normal']))
        
        return story
    
    def _create_color_legend(self, color_counts: List[int]) -> List:
        """Create color legend with swatches, symbols, and counts."""
        story = []
        
        story.append(Paragraph("Color Legend", self.styles['SectionHeader']))
        
        # Create table data
        table_data = [['Swatch', 'Symbol', 'Color Name', 'DMC Code', 'Quantity']]
        
        symbols = self.config.symbols.symbol_set
        
        for i, (color_config, count, symbol) in enumerate(zip(self.config.palette, color_counts, symbols)):
            # Color swatch will be handled in table styling
            table_data.append([
                f"[Color {i+1}]",
                symbol,
                color_config.name,
                color_config.dmc_code,
                str(count)
            ])
        
        # Create table
        table = Table(table_data, colWidths=[30*mm, 20*mm, 40*mm, 30*mm, 30*mm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), Color(0.8, 0.8, 0.8)),
            ('TEXTCOLOR', (0, 0), (-1, 0), black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), white),
            ('GRID', (0, 0), (-1, -1), 1, black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        # Add color backgrounds for swatch column
        for i, color_config in enumerate(self.config.palette):
            rgb = [c/255.0 for c in color_config.rgb]
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, i+1), (0, i+1), Color(*rgb)),
                ('TEXTCOLOR', (0, i+1), (0, i+1), white if sum(color_config.rgb) < 400 else black),
            ]))
        
        story.append(table)
        story.append(PageBreak())
        
        return story
    
    def _create_instructions(self) -> List:
        """Create assembly instructions."""
        story = []
        
        story.append(Paragraph("Assembly Instructions", self.styles['SectionHeader']))
        
        instructions = f"""
        <b>Getting Started:</b><br/><br/>
        
        1. <b>Prepare Your Canvas:</b><br/>
           • Ensure your canvas is clean and flat<br/>
           • Organize your drills/paints by color using the legend<br/>
           • Set up your workspace in a well-lit area<br/><br/>
        
        2. <b>Reading the Pattern:</b><br/>
           • Each symbol corresponds to a specific color<br/>
           • Work from top to bottom, left to right<br/>
           • Use the color legend as your reference guide<br/><br/>
        
        3. <b>Assembly Tips:</b><br/>
           • Apply a small section at a time for best results<br/>
           • Keep drills/paints sealed when not in use<br/>
           • Use the provided quantities (includes {int(self.config.output.spare_percentage*100)}% spare)<br/><br/>
        
        4. <b>Completion Time:</b><br/>
           • Estimated time: {self._estimate_completion_time()}<br/>
           • Take breaks to avoid eye strain<br/>
           • Enjoy the creative process!<br/><br/>
        
        <b>Need Help?</b><br/>
        Contact us if you have questions about your kit.
        """
        
        story.append(Paragraph(instructions, self.styles['Normal']))
        story.append(PageBreak())
        
        return story
    
    def _create_pattern_pages(self, symbol_grid: np.ndarray) -> List:
        """Create pattern pages with tiling if necessary."""
        story = []
        
        story.append(Paragraph("Pattern Grid", self.styles['SectionHeader']))
        
        # Calculate if tiling is needed
        drill_size_px = self.config.processing.dpi * self.config.canvas.drill_size_mm / 25.4  # Convert mm to pixels
        pattern_width = symbol_grid.shape[1] * drill_size_px
        pattern_height = symbol_grid.shape[0] * drill_size_px
        
        if pattern_width > self.content_width or pattern_height > self.content_height:
            # Need tiling
            story.extend(self._create_tiled_pattern(symbol_grid, drill_size_px))
        else:
            # Single page pattern
            story.extend(self._create_single_page_pattern(symbol_grid, drill_size_px))
        
        return story
    
    def _create_single_page_pattern(self, symbol_grid: np.ndarray, drill_size_px: float) -> List:
        """Create a single page pattern."""
        story = []
        
        # Create pattern image
        pattern_img = self._create_pattern_image(symbol_grid, drill_size_px)
        
        # Convert to ReportLab image
        buffer = io.BytesIO()
        pattern_img.save(buffer, format='PNG', dpi=(self.config.processing.dpi, self.config.processing.dpi))
        buffer.seek(0)
        
        rl_image = RLImage(buffer)
        
        # Scale to fit content area
        available_width = self.content_width
        available_height = self.content_height
        img_width, img_height = pattern_img.size
        
        scale = min(available_width / img_width, available_height / img_height)
        
        rl_image.drawWidth = img_width * scale
        rl_image.drawHeight = img_height * scale
        
        story.append(rl_image)
        
        return story
    
    def _create_tiled_pattern(self, symbol_grid: np.ndarray, drill_size_px: float) -> List:
        """Create tiled pattern across multiple pages."""
        story = []
        
        overlap = self.config.pdf.overlap_mm * mm
        
        # Calculate tile dimensions
        tile_width = int(self.content_width / drill_size_px)
        tile_height = int(self.content_height / drill_size_px)
        
        overlap_drills = int(overlap / drill_size_px)
        
        grid_height, grid_width = symbol_grid.shape
        
        page_num = 1
        for y_start in range(0, grid_height, tile_height - overlap_drills):
            for x_start in range(0, grid_width, tile_width - overlap_drills):
                y_end = min(y_start + tile_height, grid_height)
                x_end = min(x_start + tile_width, grid_width)
                
                # Extract tile
                tile = symbol_grid[y_start:y_end, x_start:x_end]
                
                # Create tile image
                tile_img = self._create_pattern_image(tile, drill_size_px)
                
                # Add page info
                story.append(Paragraph(f"Pattern Page {page_num}", self.styles['Heading3']))
                story.append(Paragraph(f"Position: ({x_start+1}, {y_start+1}) to ({x_end}, {y_end})", self.styles['Normal']))
                
                # Convert to ReportLab image
                buffer = io.BytesIO()
                tile_img.save(buffer, format='PNG', dpi=(self.config.processing.dpi, self.config.processing.dpi))
                buffer.seek(0)
                
                rl_image = RLImage(buffer)
                story.append(rl_image)
                
                if not (y_end >= grid_height and x_end >= grid_width):
                    story.append(PageBreak())
                
                page_num += 1
        
        return story
    
    def _create_pattern_image(self, symbol_grid: np.ndarray, drill_size_px: float) -> Image.Image:
        """Create pattern image with symbols."""
        h, w = symbol_grid.shape
        img_width = int(w * drill_size_px)
        img_height = int(h * drill_size_px)
        
        # Create white background
        img = Image.new('RGB', (img_width, img_height), (255, 255, 255))
        
        # Draw symbols (simplified - in production would use proper font rendering)
        from PIL import ImageDraw, ImageFont
        
        try:
            font = ImageFont.truetype("arial.ttf", int(drill_size_px * 0.6))
        except:
            font = ImageFont.load_default()
        
        draw = ImageDraw.Draw(img)
        
        for y in range(h):
            for x in range(w):
                symbol = symbol_grid[y, x]
                if symbol:
                    pos_x = int(x * drill_size_px + drill_size_px / 2)
                    pos_y = int(y * drill_size_px + drill_size_px / 2)
                    
                    # Draw symbol
                    draw.text((pos_x, pos_y), symbol, fill=(0, 0, 0), font=font, anchor='mm')
        
        return img
    
    def _numpy_to_pil(self, img_array: np.ndarray, size: Tuple[int, int] = None) -> Image.Image:
        """Convert numpy array to PIL Image."""
        img = Image.fromarray(img_array)
        if size:
            img = img.resize(size, Image.Resampling.LANCZOS)
        return img
    
    def _estimate_completion_time(self) -> str:
        """Estimate completion time based on canvas size and color count."""
        total_drills = (self.config.canvas.width_cm * 10 / self.config.canvas.drill_size_mm) * \
                      (self.config.canvas.height_cm * 10 / self.config.canvas.drill_size_mm)
        
        # Average time: 30 seconds per drill for beginners
        minutes = int(total_drills * 0.5)
        hours = minutes // 60
        remaining_minutes = minutes % 60
        
        if hours > 0:
            return f"~{hours}h {remaining_minutes}min"
        else:
            return f"~{minutes}min"
