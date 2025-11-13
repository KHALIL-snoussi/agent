import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diamondkit.kit_generator import generate_diamond_kit, get_style_info


def _create_demo_image(tmp_path: Path, size=(64, 64)) -> Path:
    """Create a small gradient image for deterministic testing."""
    width, height = size
    gradient = np.linspace(0, 255, num=width * height, dtype=np.uint8).reshape(height, width)
    rgb = np.stack([gradient, np.flipud(gradient), np.full_like(gradient, 180)], axis=2)
    image_path = tmp_path / "demo_input.png"
    Image.fromarray(rgb).save(image_path)
    return image_path


def test_metadata_contract_includes_required_fields(tmp_path):
    image_path = _create_demo_image(tmp_path)
    output_dir = tmp_path / "kit"
    result = generate_diamond_kit(str(image_path), "ORIGINAL", str(output_dir))
    assert result["metadata"], "Metadata should be returned in generation result"
    
    metadata_path = output_dir / "kit_metadata.json"
    assert metadata_path.exists(), "kit_metadata.json must be written"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    
    required_keys = [
        "paper_mm",
        "dpi",
        "margins_mm",
        "cell_mm",
        "grid_cols",
        "grid_rows",
        "total_cells",
        "pages",
        "pattern_pages",
        "style",
        "fixed_palette",
        "fixed_palette_dmc",
        "deltaE_stats",
        "ssim",
        "crop_rect_norm",
        "tiling_map",
        "grid_index_map_hash",
        "quality_warnings",
        "color_usage",
        "preprocessing",
        "processing_settings",
        "quality_assessment",
        "quality_gates",
    ]
    for key in required_keys:
        assert key in metadata, f"{key} missing from metadata"
    
    assert len(metadata["fixed_palette_dmc"]) == 7
    assert metadata["fixed_palette"], "Should report fixed palette constraint"
    assert metadata["total_cells"] <= 10000
    assert isinstance(metadata["tiling_map"], list) and metadata["tiling_map"]
    assert metadata["grid_index_map_hash"]
    assert isinstance(metadata["quality_warnings"], list)
    assert metadata["preprocessing"]["crop_rect_norm"] is not None
    assert "kit_metadata" in result["outputs"], "Outputs should include kit_metadata path"


def test_inventory_rows_match_fixed_palette(tmp_path):
    image_path = _create_demo_image(tmp_path)
    output_dir = tmp_path / "kit_inventory"
    generate_diamond_kit(str(image_path), "ORIGINAL", str(output_dir))
    
    palette_codes = get_style_info("ORIGINAL")["dmc_codes"]
    inventory_path = output_dir / "inventory.csv"
    assert inventory_path.exists()
    rows = inventory_path.read_text(encoding="utf-8").strip().splitlines()
    header, *data_rows = rows
    assert len(data_rows) == 7, "Inventory should list all 7 palette colors"
    codes = [line.split(",")[0] for line in data_rows]
    assert codes == palette_codes, "Inventory order should match fixed palette order"


def test_scale_factor_and_output_manifest(tmp_path):
    image_path = _create_demo_image(tmp_path, size=(48, 48))
    output_dir = tmp_path / "kit_manifest"
    result = generate_diamond_kit(str(image_path), "ORIGINAL", str(output_dir))
    metadata = result["metadata"]
    
    # Scale factor recorded in metadata should match runtime result (rounded to 3 decimals)
    assert "scale_factor" in metadata
    assert abs(metadata["scale_factor"] - round(result["scale_factor"], 3)) < 1e-3
    
    # Outputs should report the metadata filename so UI layers can resolve downloads
    output_manifest = metadata["output_files"]
    assert "kit_metadata" in output_manifest
    assert output_manifest["kit_metadata"] == os.path.basename(result["outputs"]["kit_metadata"])
    
    # Quality warnings/risk arrays must always exist for CLI/web consumption
    assert isinstance(metadata.get("quality_warnings"), list)
    assert isinstance(metadata.get("quality_risks"), list)
