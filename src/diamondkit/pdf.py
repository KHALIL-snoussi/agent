"""
PDF generation for QBRIX-style diamond painting instruction booklets.
Fixed 7-color DMC palettes with numeric grids and professional assembly instructions.
"""

import os
import math
from typing import List, Tuple, Dict, Any
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import Color, black, white, lightgrey
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.platypus import PageBreak, KeepTogether
from reportlab.pdfgen import canvas
from PIL import Image
import numpy as np

from .grid_index_map import GridIndexMap
from .print_math import PrintSpecs, PrintMathEngine


class QBRIXPDFGenerator:
    """Generates QBRIX-quality PDF instruction booklets with fixed 7-color palettes."""
    
    def __init__(self, print_specs: PrintSpecs):
        """Initialize PDF generator with print specifications."""
        self.print_specs = print_specs
        self.print_engine = PrintMathEngine(print_specs)
        
        # Page setup (always A4 landscape for QBRIX)
        self.page_size = landscape(A4)
        self.page_w, self.page_h = self.page_size
        self.dpi = print_specs.dpi
        
        # Cell size calculations
        self.cell_size_mm = print_specs.cell_size_mm
        self.cell_size_pt = self.cell_size_mm * mm
        
        # Margins and usable area
        self.margin_mm = print_specs.margin_mm
        self.margin_pt = self.margin_mm * mm
        self.usable_w = self.page_w - 2 * self.margin_pt
        self.usable_h = self.page_h - 2 * self.margin_pt
        
        # Symbol font settings (digits 1-7)
        self.symbol_font = "Helvetica-Bold"
        self.symbol_font_size_pt = self.cell_size_pt * 0.4  # 40% of cell size
        
        # Verify symbol legibility
        self.x_height_mm = self.cell_size_mm * 0.43  # Approximate x-height for digits
        self.stroke_thickness_mm = self.cell_size_mm * 0.08  # 8% of cell size
    
    def generate_qbrix_pdf(self, grid_map: GridIndexMap, 
                          preview_image: np.ndarray,
                          metadata: Dict[str, Any],
                          output_path: str) -> str:
        """
        Generate complete QBRIX-style PDF instruction booklet.
        
        Args:
            grid_map: Grid index map with fixed palette
            preview_image: Preview of completed design
            metadata: Kit metadata
            output_path: Output PDF path
            
        Returns:
            Path to generated PDF
        """
        print(f"Generating QBRIX PDF with fixed 7-color palette...")
        
        # Calculate tiling for this grid
        tiles = self.print_engine.calculate_tiling(grid_map.grid_specs)
        
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
        
        # Page 1: Title/overview
        story.extend(self._create_qbrix_title_page(grid_map, preview_image, metadata))
        story.append(PageBreak())
        
        # Page 2: Professional legend with 7 fixed colors
        story.extend(self._create_qbrix_legend_page(grid_map))
        story.append(PageBreak())
        
        # Page 3: Assembly instructions
        story.extend(self._create_qbrix_instructions_page(grid_map, tiles))
        story.append(PageBreak())
        
        # Pages 4+: Numeric grid tiles
        story.extend(self._create_qbrix_pattern_pages(grid_map, tiles))
        
        # Build PDF
        doc.build(story)
        print(f"QBRIX PDF generated: {output_path}")
        return output_path
    
    def _create_qbrix_title_page(self, grid_map: GridIndexMap,
                               preview_image: np.ndarray,
                               metadata: Dict[str, Any]) -> List:
        """Create QBRIX title page with prominent preview image and professional layout."""
        from reportlab.platypus import Image as RLImage
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'QBRIXTitle',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=20,
            alignment=1  # Center
        )
        
        heading_style = ParagraphStyle(
            'QBRIXHeading',
            parent=styles['Heading1'],
            fontSize=14,
            spaceAfter=8
        )
        
        normal_style = ParagraphStyle(
            'QBRIXNormal',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=6
        )
        
        content = []
        
        # Main title
        content.append(Paragraph("QBRIX DIAMOND PAINTING KIT", title_style))
        content.append(Spacer(1, 10))
        
        # PROMINENT PREVIEW IMAGE - QBRIX-style
        try:
            # Convert numpy array to PIL Image
            pil_preview = Image.fromarray(preview_image)
            
            # Calculate optimal preview size (60-70% of usable width)
            max_preview_width = self.usable_w * 0.65
            max_preview_height = self.usable_h * 0.35
            
            # Maintain aspect ratio
            img_aspect = pil_preview.width / pil_preview.height
            if max_preview_width / max_preview_height > img_aspect:
                # Height limited
                preview_height = max_preview_height
                preview_width = preview_height * img_aspect
            else:
                # Width limited
                preview_width = max_preview_width
                preview_height = preview_width / img_aspect
            
            # Convert to ReportLab Image
            rl_image = RLImage(pil_preview, 
                              width=preview_width, 
                              height=preview_height)
            
            # Center the image
            image_container = KeepTogether([
                Spacer(1, 5),
                rl_image,
                Spacer(1, 15)
            ])
            
            content.append(image_container)
            print(f"Added prominent preview image: {preview_width:.0f}×{preview_height:.0f} points")
            
        except Exception as e:
            print(f"Warning: Could not add preview image to title page: {e}")
            content.append(Spacer(1, 20))
        
        # Style and palette info (below preview)
        content.append(Paragraph(f"<b>Style:</b> {grid_map.style_name.upper()} | <b>Fixed 7-Color DMC Palette</b>", heading_style))
        content.append(Spacer(1, 10))
        
        # COMPACT SPECIFICATIONS TABLE - Two columns for better layout
        specs_left = [
            ['Specification', 'Value'],
            ['Grid Size', f"{grid_map.grid_specs.cols} × {grid_map.grid_specs.rows}"],
            ['Total Cells', f"{grid_map.grid_specs.total_cells:,}"],
            ['Cell Size', f"{self.cell_size_mm:.1f} mm"],
            ['Print DPI', f"{self.dpi}"],
            ['Paper Size', 'A4 Landscape']
        ]
        
        specs_right = [
            ['Specification', 'Value'],
            ['Margins', f"{self.margin_mm:.0f} mm"],
            ['Total Pages', f"{len(self.print_engine.calculate_tiling(grid_map.grid_specs)) + 3}"],
            ['Colors Used', f"{len(np.unique(grid_map.grid_data))}/7"],
            ['Symbol Size', f"{self.x_height_mm:.2f}mm x-height"],
            ['Grid Hash', f"{grid_map.grid_hash}"]
        ]
        
        # Create two-column table layout
        specs_left_table = Table(specs_left, colWidths=[50*mm, 60*mm])
        specs_left_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), lightgrey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 1, black),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        specs_right_table = Table(specs_right, colWidths=[50*mm, 60*mm])
        specs_right_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), lightgrey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 1, black),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        # Combine tables side by side
        combined_specs_data = [
            [specs_left_table, specs_right_table]
        ]
        combined_specs_table = Table(combined_specs_data, colWidths=[120*mm, 120*mm])
        combined_specs_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        content.append(combined_specs_table)
        content.append(Spacer(1, 15))
        
        # Quality metrics section
        content.append(Paragraph("Quality Assessment", heading_style))
        
        deltaE_mean = metadata.get('deltaE_stats', {}).get('mean', 'N/A')
        deltaE_max = metadata.get('deltaE_stats', {}).get('max', 'N/A')
        ssim = metadata.get('ssim', 'N/A')
        scale_factor = metadata.get('scale_factor', 1.0)
        
        quality_data = [
            ['Metric', 'Value', 'Status'],
            ['ΔE2000 Mean', f"{deltaE_mean}" if deltaE_mean != 'N/A' else 'N/A', 
             '✓ Good' if isinstance(deltaE_mean, (int, float)) and deltaE_mean < 40 else '⚠ Check'],
            ['ΔE2000 Max', f"{deltaE_max}" if deltaE_max != 'N/A' else 'N/A',
             '✓ Good' if isinstance(deltaE_max, (int, float)) and deltaE_max < 60 else '⚠ Check'],
            ['SSIM Score', f"{ssim}" if ssim != 'N/A' else 'N/A',
             '✓ Good' if isinstance(ssim, (int, float)) and ssim > 0.6 else '⚠ Check'],
            ['Scale Factor', f"{scale_factor:.2f}" if scale_factor != 1.0 else '1.00', 'Normal']
        ]
        
        quality_table = Table(quality_data, colWidths=[40*mm, 40*mm, 35*mm])
        quality_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), lightgrey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, black),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        content.append(quality_table)
        
        # Warnings section (if any)
        warnings = metadata.get('warnings', [])
        if warnings:
            content.append(Spacer(1, 10))
            content.append(Paragraph("⚠ Quality Warnings", heading_style))
            for warning in warnings[:3]:  # Limit to first 3 warnings
                content.append(Paragraph(f"• {warning}", normal_style))
        
        # Assembly tips footer
        content.append(Spacer(1, 15))
        content.append(Paragraph("<b>Quick Start:</b> Follow the color legend (pages 2-3) and use the numeric pattern pages to place diamonds. Match symbols 1-7 to your DMC drills.", normal_style))
        
        return content
    
    def _create_qbrix_legend_page(self, grid_map: GridIndexMap) -> List:
        """Create professional legend page for 7 fixed DMC colors."""
        styles = getSampleStyleSheet()
        heading_style = ParagraphStyle(
            'QBRIXLegend',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=20,
            alignment=1  # Center
        )
        
        content = []
        content.append(Paragraph("COLOR LEGEND - FIXED PALETTE", heading_style))
        content.append(Paragraph(f"Style: {grid_map.style_name.upper()}", styles['Normal']))
        content.append(Spacer(1, 15))
        
        # Count color usage
        h, w = grid_map.grid_data.shape
        unique_indices, counts = np.unique(grid_map.grid_data, return_counts=True)
        
        # Create legend data
        legend_data = [['Symbol', 'DMC Code', 'Color Name', 'Hex', 'Drill Count', 'Bag Qty (200)']]
        
        for idx, count in zip(unique_indices, counts):
            if idx < len(grid_map.palette_colors):
                color = grid_map.palette_colors[idx]
                symbol = str(idx + 1)  # Use 1-7 instead of 0-6 for user-friendliness
                bag_qty = -(-count // 200)  # Ceiling division
                
                legend_data.append([
                    symbol,
                    color.dmc_code,
                    color.name,
                    color.hex,
                    f"{count:,}",
                    str(bag_qty)
                ])
        
        # Create table
        col_widths = [20*mm, 25*mm, 50*mm, 25*mm, 30*mm, 25*mm]
        legend_table = Table(legend_data, colWidths=col_widths, repeatRows=1)
        
        # Style table with professional appearance
        legend_table.setStyle(TableStyle([
            # Header
            ('BACKGROUND', (0, 0), (-1, 0), lightgrey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            
            # Data rows
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            
            # Grid
            ('GRID', (0, 0), (-1, -1), 1, black),
            
            # Color sample column (column 0 for symbols)
            ('BACKGROUND', (0, 1), (0, -1), white),
            
            # Alternating row colors for readability
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, lightgrey]),
        ]))
        
        content.append(legend_table)
        content.append(Spacer(1, 20))
        
        # Summary statistics
        total_drills = grid_map.grid_specs.total_cells
        total_bags = sum(-(-count // 200) for count in counts)
        
        summary_text = f"""
        <b>Summary:</b><br/>
        • Style: {grid_map.style_name.upper()}<br/>
        • Total Colors: 7 (fixed palette)<br/>
        • Colors Used: {len(unique_indices)}<br/>
        • Total Drills: {total_drills:,}<br/>
        • Total Bags (200 drills each): {total_bags}<br/>
        • Grid Hash: {grid_map.grid_hash}<br/>
        • Cell Size: {self.cell_size_mm:.1f}mm<br/>
        • Print Quality: {self.dpi} DPI
        """
        
        content.append(Paragraph(summary_text, styles['Normal']))
        
        return content
    
    def _create_qbrix_instructions_page(self, grid_map: GridIndexMap, 
                                    tiles: List) -> List:
        """Create QBRIX assembly instructions with tiling information."""
        styles = getSampleStyleSheet()
        heading_style = ParagraphStyle(
            'QBRIXInstructions',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=15
        )
        
        content = []
        content.append(Paragraph("ASSEMBLY INSTRUCTIONS", heading_style))
        
        # Kit-specific information
        kit_info = [
            f"<b>Kit Specifications:</b>",
            f"• Style: {grid_map.style_name.upper()} with 7 fixed DMC colors",
            f"• Grid size: {grid_map.grid_specs.cols} × {grid_map.grid_specs.rows} ({grid_map.grid_specs.total_cells:,} cells)",
            f"• Cell size: {self.cell_size_mm:.1f}mm (symbol x-height: {self.x_height_mm:.2f}mm)",
            f"• Print settings: {self.dpi} DPI, {self.margin_mm:.0f}mm margins",
            f"• Total pattern pages: {len(tiles)}",
            "",
            "<b>Printing Instructions:</b>",
            "• Print at 100% scale - do not fit or scale to page",
            "• Use A4 paper in landscape orientation",
            "• Ensure print quality is set to high (300+ DPI)",
            "• Check that crop marks and registration crosses are visible",
            "",
            "<b>Page Assembly:</b>",
            f"• This kit contains {len(tiles)} pattern pages plus reference pages",
            "• Each pattern page shows a section of the complete grid",
            "• Pages use numeric symbols 1-7 corresponding to the legend",
            "• Coordinate labels help with page alignment",
            "",
            "<b>Cutting and Alignment:</b>",
            "• Cut pages along the outer crop marks",
            "• Use registration crosses to align adjacent pages",
            f"• Pages may have {self.print_specs.overlap_cells} cell overlaps for easier matching",
            "• Trim overlaps as needed for your assembly method",
            "",
            "<b>Diamond Placement:</b>",
            "• Work with one color at a time using the legend",
            "• Match symbols on pages to symbols on your adhesive canvas",
            "• Use the coordinate system to track your progress",
            "• Symbols 1-7 correspond to the fixed DMC palette colors",
            "",
            "<b>Tips for Best Results:</b>",
            "• Work from top to bottom, left to right",
            "• Keep the protective film on unused areas",
            "• Use a roller or flat object to press down placed diamonds",
            "• Take breaks to maintain accuracy and reduce eye strain",
            "• Store unused diamonds in their original bags"
        ]
        
        for instruction in kit_info:
            if instruction.startswith("<b>"):
                content.append(Paragraph(instruction, heading_style))
            else:
                content.append(Paragraph(instruction, styles['Normal']))
            
            if instruction == "":
                content.append(Spacer(1, 6))
        
        return content
    
    def _create_qbrix_pattern_pages(self, grid_map: GridIndexMap, 
                                  tiles: List) -> List:
        """Create QBRIX numeric grid pattern pages."""
        from reportlab.platypus import Flowable
        
        class QBRIXGridPage(Flowable):
            """Custom flowable for QBRIX grid pages."""
            
            def __init__(self, grid_map, tile, pdf_generator):
                super().__init__()
                self.grid_map = grid_map
                self.tile = tile
                self.pdf_gen = pdf_generator
                # Use exact page dimensions to avoid overflow
                self.width = pdf_generator.page_w - 2 * pdf_generator.margin_pt
                self.height = pdf_generator.page_h - 2 * pdf_generator.margin_pt
                
            def wrap(self, availWidth, availHeight):
                # Ensure we fit within available space
                self.width = min(availWidth, self.width)
                self.height = min(availHeight, self.height)
                return (self.width, self.height)
            
            def draw(self):
                canvas = self.canv
                pdf_gen = self.pdf_gen
                
                # Save canvas state
                canvas.saveState()
                
                # Add QBRIX page header
                pdf_gen._add_qbrix_page_header(canvas, self.tile)
                
                # Add crop marks and registration crosses
                pdf_gen._add_qbrix_crop_marks(canvas)
                
                # Draw numeric grid
                pdf_gen._draw_qbrix_numeric_grid(canvas, self.grid_map, self.tile)
                
                # Add coordinate labels
                pdf_gen._add_qbrix_coordinate_labels(canvas, self.tile)
                
                # Add mini-map showing tile position
                pdf_gen._add_qbrix_minimap(canvas, self.grid_map, self.tile)
                
                # Restore canvas state
                canvas.restoreState()
        
        content = []
        
        for i, tile in enumerate(tiles):
            # Add custom flowable for each grid page
            grid_page = QBRIXGridPage(grid_map, tile, self)
            content.append(grid_page)
            
            # Add page break except for last page
            if i < len(tiles) - 1:
                content.append(PageBreak())
        
        return content
    
    def _add_qbrix_page_header(self, pdf_canvas, tile):
        """Add QBRIX page header with tile information."""
        pdf_canvas.setFont("Helvetica-Bold", 12)
        pdf_canvas.drawString(
            self.margin_pt, 
            self.page_h - self.margin_pt - 15,
            f"Assembly Page {tile.page_number} of {tile.total_pages}"
        )
        
        pdf_canvas.setFont("Helvetica", 9)
        pdf_canvas.drawString(
            self.margin_pt,
            self.page_h - self.margin_pt - 30,
            f"QBRIX Kit - Fixed 7-Color Palette"
        )
        
        # Tile coordinates
        coord_text = f"C{tile.x_start+1}-{tile.x_start+tile.tile_cols} : R{tile.y_start+1}-{tile.y_start+tile.tile_rows}"
        pdf_canvas.drawString(
            self.page_w - self.margin_pt - pdf_canvas.stringWidth(coord_text, "Helvetica", 9),
            self.page_h - self.margin_pt - 30,
            coord_text
        )
    
    def _add_qbrix_crop_marks(self, pdf_canvas):
        """Add QBRIX crop marks and registration crosses."""
        crop_length = 8 * mm  # 8mm crop marks
        cross_size = 3 * mm   # 3mm registration crosses
        
        # Crop marks at corners
        corners = [
            (self.margin_pt, self.page_h - self.margin_pt, "tl"),
            (self.page_w - self.margin_pt, self.page_h - self.margin_pt, "tr"),
            (self.margin_pt, self.margin_pt, "bl"),
            (self.page_w - self.margin_pt, self.margin_pt, "br")
        ]
        
        for x, y, corner in corners:
            if corner in ["tl", "bl"]:  # Left corners
                pdf_canvas.line(x - crop_length, y, x, y)  # Horizontal
            else:  # Right corners
                pdf_canvas.line(x, y, x + crop_length, y)  # Horizontal
            
            if corner in ["tl", "tr"]:  # Top corners
                pdf_canvas.line(x, y, x, y + crop_length)  # Vertical
            else:  # Bottom corners
                pdf_canvas.line(x, y, x, y - crop_length)  # Vertical
        
        # Registration crosses at quarter points
        cross_positions = [
            (self.margin_pt + self.usable_w * 0.25, self.page_h - self.margin_pt - self.usable_h * 0.25),
            (self.page_w - self.margin_pt - self.usable_w * 0.25, self.page_h - self.margin_pt - self.usable_h * 0.25),
            (self.margin_pt + self.usable_w * 0.25, self.margin_pt + self.usable_h * 0.25),
            (self.page_w - self.margin_pt - self.usable_w * 0.25, self.margin_pt + self.usable_h * 0.25),
        ]
        
        for cx, cy in cross_positions:
            pdf_canvas.setLineWidth(0.5)
            pdf_canvas.line(cx - cross_size/2, cy, cx + cross_size/2, cy)  # Horizontal
            pdf_canvas.line(cx, cy - cross_size/2, cx, cy + cross_size/2)  # Vertical
    
    def _draw_qbrix_numeric_grid(self, pdf_canvas, grid_map: GridIndexMap, tile):
        """Draw QBRIX numeric grid with digits 1-7."""
        # Calculate tile position on page
        tile_width = tile.tile_cols * self.cell_size_pt
        tile_height = tile.tile_rows * self.cell_size_pt
        
        # Center tile on page
        start_x = (self.page_w - tile_width) / 2
        start_y = (self.page_h - tile_height) / 2
        
        # Set line width for grid
        pdf_canvas.setLineWidth(0.3)  # 0.3pt lines for clarity
        
        # Draw grid cells
        for y in range(tile.tile_rows):
            for x in range(tile.tile_cols):
                # Get grid coordinates
                grid_x = tile.x_start + x
                grid_y = tile.y_start + y
                
                if (grid_y < grid_map.grid_specs.rows and grid_x < grid_map.grid_specs.cols):
                    # Get cluster index and convert to user symbol (1-7)
                    cluster_idx = grid_map.get_cell_at(grid_x, grid_y)
                    symbol = str(cluster_idx + 1)  # Convert 0-6 to 1-7
                    color = grid_map.palette_colors[cluster_idx]
                    
                    # Calculate cell position
                    cell_x = start_x + x * self.cell_size_pt
                    cell_y = start_y + (tile.tile_rows - 1 - y) * self.cell_size_pt  # Flip Y for PDF
                    
                    # Draw cell outline only (no colored background for QBRIX-style numeric grid)
                    pdf_canvas.setFillColor(white)
                    pdf_canvas.rect(cell_x, cell_y, self.cell_size_pt, self.cell_size_pt, 
                                 fill=1, stroke=0)
                    
                    # Draw cell border
                    pdf_canvas.setStrokeColor(black)
                    pdf_canvas.setLineWidth(0.3)
                    pdf_canvas.rect(cell_x, cell_y, self.cell_size_pt, self.cell_size_pt, 
                                 fill=0, stroke=1)
                    
                    # Draw numeric symbol
                    pdf_canvas.setFillColor(black)
                    pdf_canvas.setFont(self.symbol_font, self.symbol_font_size_pt)
                    
                    # Center text in cell
                    text_width = pdf_canvas.stringWidth(symbol, self.symbol_font, self.symbol_font_size_pt)
                    text_x = cell_x + (self.cell_size_pt - text_width) / 2
                    text_y = cell_y + self.cell_size_pt * 0.3  # 30% from bottom
                    
                    pdf_canvas.drawString(text_x, text_y, symbol)
    
    def _add_qbrix_coordinate_labels(self, pdf_canvas, tile):
        """Add QBRIX coordinate labels around the grid."""
        # Calculate tile boundaries
        tile_width = tile.tile_cols * self.cell_size_pt
        tile_height = tile.tile_rows * self.cell_size_pt
        
        start_x = (self.page_w - tile_width) / 2
        start_y = (self.page_h - tile_height) / 2
        
        pdf_canvas.setFont("Helvetica", 8)
        pdf_canvas.setFillColor(black)
        
        # Column labels (top and bottom) - show every 5th column
        for x in range(0, tile.tile_cols, 5):
            col_num = tile.x_start + x + 1  # 1-based indexing
            x_pos = start_x + x * self.cell_size_pt + self.cell_size_pt / 2
            
            # Top label
            text_width = pdf_canvas.stringWidth(str(col_num), "Helvetica", 8)
            pdf_canvas.drawString(x_pos - text_width/2, start_y + tile_height + 3, str(col_num))
            
            # Bottom label
            pdf_canvas.drawString(x_pos - text_width/2, start_y - 8, str(col_num))
        
        # Row labels (left and right) - show every 5th row
        for y in range(0, tile.tile_rows, 5):
            row_num = tile.y_start + y + 1  # 1-based indexing
            y_pos = start_y + (tile.tile_rows - 1 - y) * self.cell_size_pt + self.cell_size_pt / 2
            
            text_width = pdf_canvas.stringWidth(str(row_num), "Helvetica", 8)
            
            # Left label
            pdf_canvas.drawString(start_y - text_width - 5, y_pos, str(row_num))
            
            # Right label
            pdf_canvas.drawString(start_x + tile_width + 5, y_pos, str(row_num))
    
    def _add_qbrix_minimap(self, pdf_canvas, grid_map: GridIndexMap, tile):
        """Add mini-map showing tile position in full grid."""
        # Mini-map dimensions
        map_w = 30 * mm
        map_h = 20 * mm
        map_x = self.page_w - self.margin_pt - map_w - 5
        map_y = self.margin_pt + 5
        
        # Draw mini-map border
        pdf_canvas.setLineWidth(0.5)
        pdf_canvas.rect(map_x, map_y, map_w, map_h, fill=0, stroke=1)
        
        # Calculate scale factors
        scale_x = map_w / grid_map.grid_specs.cols
        scale_y = map_h / grid_map.grid_specs.rows
        
        # Draw current tile highlight
        tile_x = map_x + tile.x_start * scale_x
        tile_y = map_y + (grid_map.grid_specs.rows - tile.y_start - tile.tile_rows) * scale_y
        tile_w = tile.tile_cols * scale_x
        tile_h = tile.tile_rows * scale_y
        
        pdf_canvas.setFillColor(lightgrey)
        pdf_canvas.rect(tile_x, tile_y, tile_w, tile_h, fill=1, stroke=1)
        
        # Add label
        pdf_canvas.setFont("Helvetica", 7)
        pdf_canvas.setFillColor(black)
        label_text = "Position"
        text_width = pdf_canvas.stringWidth(label_text, "Helvetica", 7)
        pdf_canvas.drawString(map_x + map_w/2 - text_width/2, map_y + map_h + 3, label_text)
