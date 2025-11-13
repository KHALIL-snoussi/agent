"""
High-fidelity preprocessing for QBRIX diamond painting kits.
Handles smart cropping, exposure control, denoising, gamma, and grid-ready resampling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PIL import Image, ImageOps
import cv2
import yaml
from skimage.color import rgb2lab

from .print_math import PrintMathEngine, GridSpecs


@dataclass
class ProcessingSettings:
    """Configurable preprocessing and quantization settings."""

    long_side_px: int = 1200
    min_long_side_px: int = 900
    exposure_clip_percent: float = 0.8
    gamma_low: float = 0.9
    gamma_high: float = 1.1
    denoise_sigma_color: float = 35.0
    denoise_sigma_space: float = 5.0
    enable_saliency_crop: bool = True
    enable_face_detection: bool = True
    saliency_threshold_percentile: float = 75.0
    crop_aspect_tolerance: float = 0.02
    smoothing_enabled: bool = True
    smoothing_kernel: int = 3
    popart_edge_bias: float = 18.0
    popart_edge_threshold: float = 0.25


@dataclass
class PreprocessResult:
    """Container for preprocessing outputs used downstream."""

    original_rgb: np.ndarray
    cropped_rgb: np.ndarray
    working_rgb: np.ndarray
    grid_rgb: np.ndarray
    grid_lab: np.ndarray
    crop_rect_norm: Tuple[float, float, float, float]
    grid_specs: GridSpecs
    scale_factor: float
    face_bboxes_norm: List[Tuple[float, float, float, float]]
    saliency_center_norm: Tuple[float, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_processing_settings(config_path: str = "config.yaml") -> ProcessingSettings:
    """Load processing overrides from config.yaml if available."""

    settings = ProcessingSettings()
    if not os.path.exists(config_path):
        return settings

    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return settings
    except Exception:
        # Fallback silently to defaults if YAML parsing fails
        return settings

    proc = data.get("processing", {})
    smoothing_cfg = proc.get("smoothing", {})
    popart_cfg = proc.get("popart", {})

    return ProcessingSettings(
        long_side_px=int(proc.get("long_side_px", settings.long_side_px)),
        min_long_side_px=int(proc.get("min_long_side_px", settings.min_long_side_px)),
        exposure_clip_percent=float(
            proc.get("exposure_clip_percent", settings.exposure_clip_percent)
        ),
        gamma_low=float(proc.get("gamma_range", [settings.gamma_low, settings.gamma_high])[0]),
        gamma_high=float(proc.get("gamma_range", [settings.gamma_low, settings.gamma_high])[-1]),
        denoise_sigma_color=float(
            proc.get("denoise_sigma_color", settings.denoise_sigma_color)
        ),
        denoise_sigma_space=float(
            proc.get("denoise_sigma_space", settings.denoise_sigma_space)
        ),
        enable_saliency_crop=bool(proc.get("auto_crop", settings.enable_saliency_crop)),
        enable_face_detection=bool(
            proc.get("enable_face_detection", settings.enable_face_detection)
        ),
        saliency_threshold_percentile=float(
            proc.get(
                "saliency_threshold_percentile",
                settings.saliency_threshold_percentile,
            )
        ),
        crop_aspect_tolerance=float(
            proc.get("crop_aspect_tolerance", settings.crop_aspect_tolerance)
        ),
        smoothing_enabled=bool(smoothing_cfg.get("enabled", settings.smoothing_enabled)),
        smoothing_kernel=int(smoothing_cfg.get("kernel", settings.smoothing_kernel)),
        popart_edge_bias=float(popart_cfg.get("edge_bias", settings.popart_edge_bias)),
        popart_edge_threshold=float(
            popart_cfg.get("edge_threshold", settings.popart_edge_threshold)
        ),
    )


class ImagePreprocessor:
    """Content-aware preprocessing pipeline that feeds the fixed palette quantizer."""

    def __init__(self, print_engine: PrintMathEngine, settings: ProcessingSettings):
        self.print_engine = print_engine
        self.settings = settings
        self._face_detector: Optional[cv2.CascadeClassifier] = None

    def process_image(
        self,
        image_path: str,
        crop_rect: Optional[Tuple[float, float, float, float]] = None,
    ) -> PreprocessResult:
        """Load, crop, enhance, and resize image for quantization."""

        original_rgb = self._load_image(image_path)

        if crop_rect:
            cropped_rgb, crop_norm = self._crop_with_rect(original_rgb, crop_rect)
            faces_norm: List[Tuple[float, float, float, float]] = []
            saliency_center_norm = (0.5, 0.5)
        else:
            cropped_rgb, crop_norm, faces_norm, saliency_center_norm = self._auto_crop(
                original_rgb
            )

        working_rgb = self._resize_long_side(cropped_rgb)
        adjusted_rgb, adjustments_meta = self._apply_enhancements(working_rgb)

        grid_specs, scale_factor = self.print_engine.calculate_grid_from_image(
            cropped_rgb.shape[1], cropped_rgb.shape[0]
        )

        grid_rgb = self._resize_to_grid(adjusted_rgb, grid_specs)
        grid_lab = rgb2lab(grid_rgb / 255.0).astype(np.float32)

        metadata = {
            "original_size": [int(original_rgb.shape[1]), int(original_rgb.shape[0])],
            "cropped_size": [int(cropped_rgb.shape[1]), int(cropped_rgb.shape[0])],
            "working_size": [int(working_rgb.shape[1]), int(working_rgb.shape[0])],
            "grid_size": [grid_specs.cols, grid_specs.rows],
            "crop_rect_norm": crop_norm,
            "saliency_center_norm": saliency_center_norm,
            "face_count": len(faces_norm),
            "faces_norm": faces_norm,
            "adjustments": adjustments_meta,
        }

        return PreprocessResult(
            original_rgb=original_rgb,
            cropped_rgb=cropped_rgb,
            working_rgb=working_rgb,
            grid_rgb=grid_rgb,
            grid_lab=grid_lab,
            crop_rect_norm=crop_norm,
            grid_specs=grid_specs,
            scale_factor=scale_factor,
            face_bboxes_norm=faces_norm,
            saliency_center_norm=saliency_center_norm,
            metadata=metadata,
        )

    @staticmethod
    def _load_image(image_path: str) -> np.ndarray:
        """Load image with EXIF-aware orientation."""

        pil_image = Image.open(image_path)
        pil_image = ImageOps.exif_transpose(pil_image)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        return np.array(pil_image)

    def _crop_with_rect(
        self, image_rgb: np.ndarray, rect_norm: Tuple[float, float, float, float]
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """Crop image using normalized rectangle."""

        h, w = image_rgb.shape[:2]
        x_norm, y_norm, width_norm, height_norm = rect_norm
        x_px = int(np.clip(x_norm * w, 0, w - 1))
        y_px = int(np.clip(y_norm * h, 0, h - 1))
        width_px = max(1, int(width_norm * w))
        height_px = max(1, int(height_norm * h))
        width_px = min(width_px, w - x_px)
        height_px = min(height_px, h - y_px)
        cropped = image_rgb[y_px : y_px + height_px, x_px : x_px + width_px]
        return cropped, (
            x_px / w,
            y_px / h,
            width_px / w,
            height_px / h,
        )

    def _auto_crop(
        self, image_rgb: np.ndarray
    ) -> Tuple[
        np.ndarray,
        Tuple[float, float, float, float],
        List[Tuple[float, float, float, float]],
        Tuple[float, float],
    ]:
        """Suggest a content-aware crop anchored around saliency or faces."""
        if not self.settings.enable_saliency_crop:
            h, w = image_rgb.shape[:2]
            crop_norm = (0.0, 0.0, 1.0, 1.0)
            return image_rgb, crop_norm, [], (0.5, 0.5)
        
        h, w = image_rgb.shape[:2]
        saliency_map = self._compute_saliency(image_rgb)
        saliency_sum = saliency_map.sum()

        if saliency_sum > 0:
            y_indices, x_indices = np.indices(saliency_map.shape)
            focus_x = float((x_indices * saliency_map).sum() / saliency_sum)
            focus_y = float((y_indices * saliency_map).sum() / saliency_sum)
        else:
            focus_x = w / 2
            focus_y = h / 2

        faces_norm = self._detect_faces(image_rgb) if self.settings.enable_face_detection else []
        if faces_norm:
            # Use the centroid of detected faces
            centers = [
                (face[0] + face[2] / 2.0, face[1] + face[3] / 2.0)
                for face in faces_norm
            ]
            focus_x = np.mean([c[0] * w for c in centers])
            focus_y = np.mean([c[1] * h for c in centers])

        bounding_mask = saliency_map >= np.percentile(
            saliency_map, self.settings.saliency_threshold_percentile
        )
        if np.count_nonzero(bounding_mask) < 20:
            bounding_mask = saliency_map >= saliency_map.max()

        if np.count_nonzero(bounding_mask) >= 20:
            ys, xs = np.where(bounding_mask)
            min_x, max_x = xs.min(), xs.max()
            min_y, max_y = ys.min(), ys.max()
        else:
            min_x, min_y, max_x, max_y = 0, 0, w - 1, h - 1

        bbox_w = max_x - min_x + 1
        bbox_h = max_y - min_y + 1
        target_ratio = w / h if h else 1.0

        desired_w = min(w, max(int(bbox_w * 1.35), int(w * 0.4)))
        desired_h = int(desired_w / target_ratio) if target_ratio else bbox_h
        if desired_h > h:
            desired_h = h
            desired_w = int(desired_h * target_ratio)

        cx = np.clip(focus_x, desired_w / 2, w - desired_w / 2)
        cy = np.clip(focus_y, desired_h / 2, h - desired_h / 2)
        x0 = int(round(cx - desired_w / 2))
        y0 = int(round(cy - desired_h / 2))
        x0 = max(0, min(x0, w - desired_w))
        y0 = max(0, min(y0, h - desired_h))

        cropped = image_rgb[y0 : y0 + desired_h, x0 : x0 + desired_w]
        crop_norm = (x0 / w, y0 / h, desired_w / w, desired_h / h)
        saliency_center_norm = (cx / w, cy / h)

        return cropped, crop_norm, faces_norm, saliency_center_norm

    @staticmethod
    def _compute_saliency(image_rgb: np.ndarray) -> np.ndarray:
        """Approximate saliency using edge density and local variance."""

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = cv2.magnitude(grad_x, grad_y)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        local_var = cv2.GaussianBlur(laplacian ** 2, (0, 0), 3)

        edge_norm = cv2.normalize(edge_mag, None, 0.0, 1.0, cv2.NORM_MINMAX)
        var_norm = cv2.normalize(np.abs(local_var), None, 0.0, 1.0, cv2.NORM_MINMAX)
        saliency = np.clip(0.65 * edge_norm + 0.35 * var_norm, 0.0, 1.0)
        return saliency

    def _detect_faces(
        self, image_rgb: np.ndarray
    ) -> List[Tuple[float, float, float, float]]:
        """Detect faces using Haar cascades (best-effort)."""

        if self._face_detector is None:
            cascade_path = getattr(cv2.data, "haarcascades", "")
            cascade_file = os.path.join(cascade_path, "haarcascade_frontalface_default.xml")
            if os.path.exists(cascade_file):
                self._face_detector = cv2.CascadeClassifier(cascade_file)
            else:
                return []

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        faces = self._face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(40, 40),
        )

        h, w = gray.shape[:2]
        faces_norm = [
            (x / w, y / h, width / w, height / h) for x, y, width, height in faces
        ]
        return faces_norm

    def _resize_long_side(self, image_rgb: np.ndarray) -> np.ndarray:
        """Resize image so the long side fits within configured bounds."""

        h, w = image_rgb.shape[:2]
        long_side = max(h, w)
        target = self.settings.long_side_px
        min_target = self.settings.min_long_side_px

        if long_side > target:
            scale = target / long_side
        elif long_side < min_target:
            scale = min_target / long_side
        else:
            return image_rgb.copy()

        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized

    def _apply_enhancements(self, working_rgb: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply exposure normalization, white balance, denoise, and gamma."""

        rgb = working_rgb.astype(np.float32) / 255.0
        clip = max(0.0, min(self.settings.exposure_clip_percent, 5.0))
        adjustments: Dict[str, Any] = {}

        if clip > 0:
            lower = np.percentile(rgb, clip, axis=(0, 1))
            upper = np.percentile(rgb, 100 - clip, axis=(0, 1))
            scale = np.where(upper - lower > 1e-3, upper - lower, 1.0)
            rgb = np.clip((rgb - lower) / scale, 0.0, 1.0)
            adjustments["exposure_clip_percent"] = clip

        channel_means = rgb.reshape(-1, 3).mean(axis=0) + 1e-6
        gray_mean = channel_means.mean()
        gains = gray_mean / channel_means
        rgb = np.clip(rgb * gains, 0.0, 1.0)
        adjustments["white_balance_gains"] = gains.tolist()

        # Bilateral filter in 8-bit space for gentle denoise
        rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        denoised = cv2.bilateralFilter(
            rgb_uint8,
            d=5,
            sigmaColor=self.settings.denoise_sigma_color,
            sigmaSpace=self.settings.denoise_sigma_space,
        )
        rgb = denoised.astype(np.float32) / 255.0

        gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
        mean_luma = float(np.mean(gray))
        if mean_luma < 0.45:
            gamma = self.settings.gamma_low
        elif mean_luma > 0.65:
            gamma = self.settings.gamma_high
        else:
            gamma = 1.0
        adjustments["gamma"] = gamma

        if abs(gamma - 1.0) > 1e-3:
            rgb = np.clip(np.power(rgb, 1.0 / gamma), 0.0, 1.0)

        adjusted_rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        return adjusted_rgb, adjustments

    @staticmethod
    def _resize_to_grid(image_rgb: np.ndarray, grid_specs: GridSpecs) -> np.ndarray:
        """Resize enhanced image down to the grid resolution."""

        pil_image = Image.fromarray(image_rgb)
        resized = pil_image.resize((grid_specs.cols, grid_specs.rows), Image.LANCZOS)
        return np.array(resized, dtype=np.uint8)
