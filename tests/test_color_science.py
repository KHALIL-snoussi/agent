import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diamondkit.color_math import delta_e2000, rgb_to_lab
from diamondkit.dmc import get_dmc_registry
from diamondkit.grid_index_map import create_grid_index_map
from diamondkit.print_math import GridSpecs


def test_delta_e2000_reference_pairs():
    # Data from Sharma et al. 2005 (official CIEDE2000 examples)
    reference_pairs = [
        ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
        ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
        ((50.0, 2.8361, -74.02), (50.0, 0.0, -82.7485), 3.4412),
    ]
    for lab1, lab2, expected in reference_pairs:
        delta = float(delta_e2000(np.array(lab1), np.array(lab2)))
        assert abs(delta - expected) < 1e-3


def test_reference_dmc_hues_have_expected_lab_signatures():
    registry = get_dmc_registry()
    checks = {
        "310": lambda lab: lab[0] < 5,
        "B5200": lambda lab: lab[0] > 95,
        "321": lambda lab: lab[1] > 60,
        "444": lambda lab: lab[2] > 60,
        "700": lambda lab: lab[1] < -20 and lab[2] > 15,
        "797": lambda lab: lab[2] < -20,
        "738": lambda lab: lab[0] > 75 and abs(lab[1]) < 10,
    }
    for code, predicate in checks.items():
        color = registry.get_color_by_code(code)
        assert color is not None, f"Missing DMC code {code}"
        assert predicate(color.lab), f"DMC {code} Lab {color.lab} failed hue check"


def test_primary_rgb_samples_snap_to_expected_palette_slots():
    samples = [
        ("310", (0, 0, 0)),
        ("B5200", (255, 255, 255)),
        ("321", (215, 32, 64)),
        ("444", (252, 214, 86)),
        ("700", (60, 140, 70)),
        ("797", (30, 64, 119)),
        ("738", (236, 208, 168)),
    ]
    rgb_array = np.array([rgb for _, rgb in samples], dtype=np.uint8).reshape(len(samples), 1, 3)
    lab_array = rgb_to_lab(rgb_array.reshape(-1, 3)).reshape(len(samples), 1, 3)
    grid_specs = GridSpecs(cols=1, rows=len(samples), total_cells=len(samples))
    grid_map = create_grid_index_map(
        lab_array,
        grid_specs,
        "ORIGINAL",
        enable_smoothing=False,
        image_is_preprocessed=True,
    )
    for row_idx, (expected_code, _) in enumerate(samples):
        palette_idx = grid_map.grid_data[row_idx, 0]
        assigned_code = grid_map.palette_colors[palette_idx].dmc_code
        assert assigned_code == expected_code
