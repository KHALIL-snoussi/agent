"""
DMC color registry and palette utilities.

Centralises loading of DMC codes from CSV, caches sRGB/Lab representations,
and exposes helpers for fixed 7-color palette lookups plus nearest-neighbour
queries using the CIEDE2000 distance.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .color_math import rgb_to_lab, delta_e2000


@dataclass(frozen=True)
class DMCColor:
    """Immutable record representing a single DMC swatch."""

    dmc_code: str
    name: str
    rgb: Tuple[int, int, int]
    hex: str = field(init=False)
    lab: Tuple[float, float, float] = field(init=False)

    def __post_init__(self):
        rgb = tuple(int(c) for c in self.rgb)
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "hex", f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}")
        lab = rgb_to_lab(np.array(rgb, dtype=np.float64)).reshape(3)
        object.__setattr__(self, "lab", (float(lab[0]), float(lab[1]), float(lab[2])))


class DmcRegistry:
    """Loads and caches all DMC colors for the fixed palette system."""

    def __init__(self, csv_path: Optional[str] = None):
        self.csv_path = csv_path or self._default_csv_path()
        self._colors: Dict[str, DMCColor] = {}
        self._color_list: List[DMCColor] = []
        self._lab_matrix: Optional[np.ndarray] = None
        self._load_from_csv()

    @staticmethod
    def _default_csv_path() -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        return os.path.join(project_root, "data", "dmc.csv")

    def _load_from_csv(self) -> None:
        """Populate registry from CSV and cache Lab values."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"DMC palette file not found: {self.csv_path}")

        self._colors.clear()
        self._color_list.clear()

        with open(self.csv_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if not row or not row.get("dmc"):
                    continue
                code = row["dmc"].strip()
                try:
                    r = int(row["r"])
                    g = int(row["g"])
                    b = int(row["b"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid RGB values for DMC {code}: {row}") from exc

                name = (row.get("name") or "").strip()
                color = DMCColor(code, name, (r, g, b))
                self._colors[code] = color
                self._color_list.append(color)

        if not self._color_list:
            raise ValueError("No DMC colors loaded from CSV.")

        self._lab_matrix = np.array([color.lab for color in self._color_list], dtype=np.float64)

    def list_codes(self) -> List[str]:
        return list(self._colors.keys())
    
    @property
    def colors(self) -> List[DMCColor]:
        return list(self._color_list)

    def get_color_by_code(self, dmc_code: str) -> Optional[DMCColor]:
        return self._colors.get(dmc_code)

    def get_colors(self, codes: Iterable[str]) -> List[DMCColor]:
        colors: List[DMCColor] = []
        for code in codes:
            color = self.get_color_by_code(code)
            if color is None:
                raise ValueError(f"DMC code '{code}' not found in registry.")
            colors.append(color)
        return colors

    def ensure_codes_exist(self, codes: Iterable[str]) -> None:
        missing = [code for code in codes if code not in self._colors]
        if missing:
            raise ValueError(f"DMC codes missing from registry: {missing}")

    def lab_matrix_for_codes(self, codes: Iterable[str]) -> np.ndarray:
        return np.array([self.get_color_by_code(code).lab for code in codes], dtype=np.float64)

    def nearest_color_from_lab(self, lab_color: np.ndarray) -> DMCColor:
        if self._lab_matrix is None:
            raise RuntimeError("Registry not initialised.")
        distances = delta_e2000(self._lab_matrix, lab_color)
        idx = int(np.argmin(distances))
        return self._color_list[idx]

    def nearest_color_from_rgb(self, rgb: Tuple[int, int, int]) -> DMCColor:
        lab = rgb_to_lab(np.array(rgb, dtype=np.float64))
        return self.nearest_color_from_lab(lab)


# Backwards compatibility alias for legacy imports (e.g., DMCPalette in older modules)
DMCPalette = DmcRegistry


_REGISTRY: Optional[DmcRegistry] = None


def get_dmc_registry(csv_path: Optional[str] = None) -> DmcRegistry:
    global _REGISTRY
    if _REGISTRY is None or (csv_path and os.path.abspath(csv_path) != os.path.abspath(_REGISTRY.csv_path)):
        _REGISTRY = DmcRegistry(csv_path)
    return _REGISTRY


def get_dmc_palette(csv_path: Optional[str] = None) -> DmcRegistry:
    """Alias retained for backwards compatibility."""
    return get_dmc_registry(csv_path)


def nearest_dmc(rgb: Tuple[int, int, int]) -> DMCColor:
    """Convenience helper for legacy code paths."""
    return get_dmc_registry().nearest_color_from_rgb(rgb)
