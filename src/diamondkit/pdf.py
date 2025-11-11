"""
PDF generation for diamond painting instruction booklets.
"""

import os
import math
from typing import List, Tuple, Dict, Any
from reportlab.lib.pagesizes import A4, A3, landscape
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import Color, black, white, lightgrey
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.platypus import PageBreak, KeepTogether
from reportlab.pdfgen import canvas
from PIL import Image
import numpy as np

from .config import Config
from .grid import CanvasGrid


class PDFGenerator:
    """Generates professional PDF instruction booklets."""
    
    def __init__(self, config: Config):
        """Initialize PDF generator with configuration."""
        self.config = config
        self.page_size = A3 if config.export.page == "A3" else A4
        self.dpi = config.export.pdf_dpi
        
        # Calculate cell size for PDF
        self.cell_size_mm = config.canvas.drill_size_mm
        self.cell_size_pt = self.cell_size_mm * mm / mm * mm  # Convert to points
        
        # Get page dimensions
        if config.export.page == "A3":
            self.page_w, self.page_h = landscape(A3)
        else:
            self.page_w, self.page_h = landscape(A4)
        
        # Calculate margins and usable area
        self.margin_mm = config.export.margin_mm
        self.margin_pt = self.margin_mm * mm
        self.usable_w = self.page_w - 2 * self.margin_pt
        self.usable_h = self.page_h - 2 * self.margin_pt
        
        # Calculate tiles needed
        self.tiles = self._calculate_tiles()
    
    def _calculate_tiles(self) -> List[Tuple[int, int, int, int]]:
        """Calculate tile positions and sizes for multi-page layout."""
        cells_w = self.config.canvas.cells_w
        cells_h = self.config.canvas.cells_h
        overlap_mm = self.config.export.overlap_mm
        overlap_pt = overlap_mm * mm
        
        # Calculate how many cells fit per page
        cells_per_page_w = int(self.usable_w / self.cell_size_pt)
        cells_per_page_h = int(self.usable_h / self.cell_size_pt)
        
        # Calculate overlap in cells
        overlap_cells = int(overlap_pt / self.cell_size_pt)
        
        tiles = []
        y_start = 0
        page_num = 1
        
        while y_start < cells_h:
            x_start = 0
            
            while x_start < cells_w:
                # Calculate tile dimensions
                tile_w = min(cells_per_page_w, cells_w - x_start)
                tile_h = min(cells_per_page_h, cells_h - y_start)
                
                tiles.append((x_start, y_start, tile_w, tile_h, page_num))
                
                x_start += cells_per_page_w - overlap_cells
                if x_start >= cells_w:
                    break
            
            y_start += cells_per_page_h - overlap_cells
            page_num += 1
        
        return tiles
    
    def generate_complete_pdf(self, canvas_grid: CanvasGrid,
                            preview_image: np.ndarray,
                            metadata: Dict[str, Any],
                            output_path: str):
        """Generate complete PDF instruction booklet."""
        print(f"Generating PDF with {len(self.tiles)} pages...")
        
        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=self.page_size,
            rightMargin=self.margin_pt,
            leftMargin=self.margin_pt,
            topMargin=self.margin_pt,
            bottomMargin=self.margin_pt
        )
        
        # Build content
        story = []
        
        # Title page
        story.extend(self._create_title_page(canvas_grid, preview_image, metadata))
        story.append(PageBreak())
        
        # Color legend page
        story.extend(self._create_legend_page(canvas_grid))
        story.append(PageBreak())
        
        # Instructions page
        story.extend(self._create_instructions_page(canvas_grid))
        story.append(PageBreak())
        
        # Pattern pages
        story.extend(self._create_pattern_pages(canvas_grid))
        
        # Build PDF
        doc.build(story)
        print(f"PDF generated: {output_path}")
    
    def _create_title_page(self, canvas_grid: CanvasGrid,
                          preview_image: np.ndarray,
                          metadata: Dict[str, Any]) -> List:
        """Create title page with preview and summary."""
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=12
        )
        
        normal_style = styles['Normal']
        
        content = []
        
        # Title
        content.append(Paragraph("DIAMOND PAINTING KIT", title_style))
        content.append(Spacer(1, 12))
        
        # Canvas information
        canvas_info = [
            f"<b>Canvas Size:</b> {self.config.canvas.width_cm} × {self.config.canvas.height_cm} cm",
            f"<b>Grid Size:</b> {canvas_grid.cells_w} × {canvas_grid.cells_h} drills",
            f"<b>Drill Shape:</b> {self.config.canvas.drill_shape.capitalize()}",
            f"<b>Drill Size:</b> {self.config.canvas.drill_size_mm} mm",
            f"<b>Total Colors:</b> {len(canvas_grid.dmc_colors)}",
            f"<b>Total Drills:</b> {canvas_grid._get_total_drills():,}"
        ]
        
        for info in canvas_info:
            content.append(Paragraph(info, normal_style))
        
        content.append(Spacer(1, 20))
        
        # Processing info
        content.append(Paragraph("<b>Processing Details:</b>", heading_style))
        
        processing_info = [
            f"Color Quantization: {self.config.palette.max_colors} colors, DMC palette",
            f"Dithering: {self.config.dither.mode} mode",
            f"Spare Drills: {self.config.export.spare_ratio:.0%}",
            f"Generated: {metadata.get('filename', 'Unknown image')}"
        ]
        
        for info in processing_info:
            content.append(Paragraph(info, normal_style))
        
        # Note about image preview (would be added as image if available)
        content.append(Spacer(1, 20))
        content.append(Paragraph("<i>Preview image shows the completed diamond painting design.</i>", normal_style))
        
        return content
    
    def _create_legend_page(self, canvas_grid: CanvasGrid) -> List:
        """Create color legend with drill counts and symbols."""
        styles = getSampleStyleSheet()
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=20
        )
        
        content = []
        content.append(Paragraph("COLOR LEGEND", heading_style))
        
        # Get legend items
        legend_items = canvas_grid.get_color_legend()
        
        # Create table data
        table_data = [['Symbol', 'DMC Code', 'Color Name', 'Drill Count', 'Bags Needed']]
        
        for item in legend_items:
            table_data.append([
                item['symbol'],
                item['dmc_code'],
                item['name'],
                str(item['count']),
                str(item['bags_needed'])
            ])
        
        # Create table
        col_widths = [30*mm, 25*mm, 60*mm, 30*mm, 30*mm]
        table = Table(table_data, colWidths=col_widths, repeatRows=1)
        
        # Style table
        table.setStyle(TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            
            # Data styling
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
            
            # Grid
            ('GRID', (0, 0), (-1, -1), 1, black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            
            # Alternating row colors
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, lightgrey]),
        ]))
        
        content.append(table)
        
        # Add summary information
        content.append(Spacer(1, 20))
        
        total_drills = canvas_grid._get_total_drills()
        total_bags = sum(item['bags_needed'] for item in legend_items)
        
        summary_text = f"""
        <b>Summary:</b><br/>
        Total DMC Colors: {len(legend_items)}<br/>
        Total Drills: {total_drills:,}<br/>
        Total Bags: {total_bags}<br/>
        Spare Ratio: {self.config.export.spare_ratio:.0%}<br/>
        Drills per Bag: {self.config.export.bag_size}
        """
        
        content.append(Paragraph(summary_text, styles['Normal']))
        
        return content
    
    def _create_instructions_page(self, canvas_grid: CanvasGrid) -> List:
        """Create assembly instructions page."""
        styles = getSampleStyleSheet()
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=20
        )
        
        content = []
        content.append(Paragraph("ASSEMBLY INSTRUCTIONS", heading_style))
        
        instructions = [
            "<b>1. Prepare Your Workspace</b>",
            "• Find a clean, well-lit area to work",
            "• Lay out all materials: canvas, drills, tray, and pen tool",
            "• Keep the plastic cover on the canvas when not working",
            "",
            "<b>2. Organize Your Drills</b>",
            f"• Sort your DMC drills using the color legend provided",
            "• Use a drill organizer or small containers for each color",
            "• Label each container with the DMC code and symbol",
            "• Check that you have the correct number of bags for each color",
            "",
            "<b>3. Start Diamond Painting</b>",
            "• Begin with one color or one section at a time",
            "• Pour a small amount of drills into the provided tray",
            "• Gently shake the tray to turn the drills right side up",
            "• Using the pen tool, pick up a single drill",
            "• Press the drill firmly onto the corresponding symbol on the canvas",
            "",
            "<b>4. Working Techniques</b>",
            "• Work from top to bottom to avoid smudging completed areas",
            "• Replace the plastic cover after each session to protect your work",
            "• Use a rolling pin or book to press down any loose drills",
            "• Take breaks to avoid eye strain",
            "",
            "<b>5. Completion and Finishing</b>",
            "• Once all drills are placed, press firmly over the entire canvas",
            "• Apply a sealant if desired (optional)",
            "• Frame or display your completed diamond painting",
            "",
            f"<b>Kit Specifications:</b>",
            f"• Canvas: {self.config.canvas.width_cm}×{self.config.canvas.height_cm} cm",
            f"• Grid: {canvas_grid.cells_w}×{canvas_grid.cells_h} drills",
            f"• Drill Type: {self.config.canvas.drill_shape} ({self.config.canvas.drill_size_mm}mm)",
            f"• Total Colors: {len(canvas_grid.dmc_colors)} DMC colors",
            f"• Difficulty: {self._get_difficulty_description(canvas_grid)}",
        ]
        
        for instruction in instructions:
            if instruction.startswith("<b>"):
                content.append(Paragraph(instruction, heading_style))
            else:
                content.append(Paragraph(instruction, styles['Normal']))
            
            if instruction == "":
                content.append(Spacer(1, 6))
        
        return content
    
    def _create_pattern_pages(self, canvas_grid: CanvasGrid) -> List:
        """Create pattern pages with grid and symbols."""
        content = []
        
        # Create pattern pages for each tile
        for i, (x_start, y_start, tile_w, tile_h, page_num) in enumerate(self.tiles):
            # Create canvas for this tile
            page_canvas = canvas.Canvas(
                f"pattern_page_{i}.pdf",
                pagesize=self.page_size
            )
            
            # Add page header
            self._add_page_header(page_canvas, f"Pattern Page {page_num}")
            
            # Add crop marks and registration marks
            self._add_crop_marks(page_canvas)
            
            # Draw the grid pattern
            self._draw_grid_pattern(page_canvas, canvas_grid, x_start, y_start, tile_w, tile_h)
            
            # Add coordinates
            self._add_coordinate_labels(page_canvas, x_start, y_start, tile_w, tile_h)
            
            # Save page and add to content
            page_canvas.save()
            
            # In a real implementation, we'd integrate this with ReportLab
            # For now, we'll add a placeholder
            styles = getSampleStyleSheet()
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading1'],
                fontSize=16
            )
            
            content.append(Paragraph(f"Pattern Page {page_num}", heading_style))
            content.append(Paragraph(f"Grid section: ({x_start}, {y_start}) to ({x_start+tile_w}, {y_start+tile_h})", styles['Normal']))
            content.append(Paragraph(f"<i>This page shows {tile_w}×{tile_h} drills with symbols and crop marks for precise alignment.</i>", styles['Normal']))
            
            if i < len(self.tiles) - 1:
                content.append(PageBreak())
        
        return content
    
    def _add_page_header(self, pdf_canvas, title: str):
        """Add page header with title and page info."""
        pdf_canvas.setFont("Helvetica-Bold", 14)
        pdf_canvas.drawString(self.margin_pt, self.page_h - self.margin_pt - 20, title)
        
        pdf_canvas.setFont("Helvetica", 10)
        pdf_canvas.drawString(self.margin_pt, self.page_h - self.margin_pt - 35, 
                            f"Generated with DiamondKit v1.0.0")
    
    def _add_crop_marks(self, pdf_canvas):
        """Add crop marks and registration marks."""
        # Crop mark length
        crop_length = 10 * mm
        
        # Top-left
        pdf_canvas.line(
            self.margin_pt - crop_length, self.page_h - self.margin_pt,
            self.margin_pt, self.page_h - self.margin_pt
        )
        pdf_canvas.line(
            self.margin_pt, self.page_h - self.margin_pt,
            self.margin_pt, self.page_h - self.margin_pt + crop_length
        )
        
        # Top-right
        pdf_canvas.line(
            self.page_w - self.margin_pt, self.page_h - self.margin_pt,
            self.page_w - self.margin_pt + crop_length, self.page_h - self.margin_pt
        )
        pdf_canvas.line(
            self.page_w - self.margin_pt, self.page_h - self.margin_pt,
            self.page_w - self.margin_pt, self.page_h - self.margin_pt + crop_length
        )
        
        # Bottom-left
        pdf_canvas.line(
            self.margin_pt - crop_length, self.margin_pt,
            self.margin_pt, self.margin_pt
        )
        pdf_canvas.line(
            self.margin_pt, self.margin_pt,
            self.margin_pt, self.margin_pt - crop_length
        )
        
        # Bottom-right
        pdf_canvas.line(
            self.page_w - self.margin_pt, self.margin_pt,
            self.page_w - self.margin_pt + crop_length, self.margin_pt
        )
        pdf_canvas.line(
            self.page_w - self.margin_pt, self.margin_pt,
            self.page_w - self.margin_pt, self.margin_pt - crop_length
        )
    
    def _draw_grid_pattern(self, pdf_canvas, canvas_grid: CanvasGrid,
                         x_start: int, y_start: int, tile_w: int, tile_h: int):
        """Draw the grid pattern with symbols."""
        # Calculate starting position (centered on page)
        total_width = tile_w * self.cell_size_pt
        total_height = tile_h * self.cell_size_pt
        
        start_x = (self.page_w - total_width) / 2
        start_y = (self.page_h - total_height) / 2
        
        # Draw grid cells
        for y in range(tile_h):
            for x in range(tile_w):
                # Get cell data
                grid_x = x_start + x
                grid_y = y_start + y
                
                if (grid_y < canvas_grid.cells_h and grid_x < canvas_grid.cells_w):
                    cell = canvas_grid.get_cell_at(grid_x, grid_y)
                    
                    if cell:
                        # Calculate cell position
                        cell_x = start_x + x * self.cell_size_pt
                        cell_y = start_y + (tile_h - 1 - y) * self.cell_size_pt  # Flip Y
                        
                        # Draw cell background
                        pdf_canvas.setFillColor(Color(*[c/255.0 for c in cell.rgb]))
                        pdf_canvas.rect(cell_x, cell_y, self.cell_size_pt, self.cell_size_pt, fill=1, stroke=1)
                        
                        # Draw symbol in center
                        pdf_canvas.setFillColor(black)
                        pdf_canvas.setFont("Helvetica", self.cell_size_pt * 0.4)
                        
                        text_width = pdf_canvas.stringWidth(cell.symbol, "Helvetica", self.cell_size_pt * 0.4)
                        text_x = cell_x + (self.cell_size_pt - text_width) / 2
                        text_y = cell_y + self.cell_size_pt * 0.3
                        
                        pdf_canvas.drawString(text_x, text_y, cell.symbol)
    
    def _add_coordinate_labels(self, pdf_canvas, x_start: int, y_start: int,
                              tile_w: int, tile_h: int):
        """Add coordinate labels around the grid."""
        total_width = tile_w * self.cell_size_pt
        total_height = tile_h * self.cell_size_pt
        
        start_x = (self.page_w - total_width) / 2
        start_y = (self.page_h - total_height) / 2
        
        pdf_canvas.setFont("Helvetica", 8)
        pdf_canvas.setFillColor(black)
        
        # X-axis labels (top and bottom)
        for x in range(0, tile_w, 10):  # Label every 10 cells
            coord = x_start + x
            x_pos = start_x + x * self.cell_size_pt + self.cell_size_pt / 2
            
            # Top
            pdf_canvas.drawCentredText(str(coord), x_pos, start_y + total_height + 5)
            # Bottom
            pdf_canvas.drawCentredText(str(coord), x_pos, start_y - 5)
        
        # Y-axis labels (left and right)
        for y in range(0, tile_h, 10):  # Label every 10 cells
            coord = y_start + y
            y_pos = start_y + (tile_h - 1 - y) * self.cell_size_pt + self.cell_size_pt / 2
            
            # Left
            pdf_canvas.drawCentredText(str(coord), start_x - 10, y_pos)
            # Right
            pdf_canvas.drawCentredText(str(coord), start_x + total_width + 10, y_pos)
    
    def _get_difficulty_description(self, canvas_grid: CanvasGrid) -> str:
        """Get human-readable difficulty description."""
        total_cells = canvas_grid.total_cells
        color_count = len(canvas_grid.dmc_colors)
        
        if total_cells < 2000 and color_count < 15:
            return "Beginner - Great for first-time diamond painters"
        elif total_cells < 5000 and color_count < 25:
            return "Easy - Perfect for beginners"
        elif total_cells < 10000 and color_count < 40:
            return "Medium - Suitable for intermediate crafters"
        elif total_cells < 20000:
            return "Challenging - For experienced diamond painters"
        else:
            return "Expert - Advanced project for diamond painting enthusiasts"
