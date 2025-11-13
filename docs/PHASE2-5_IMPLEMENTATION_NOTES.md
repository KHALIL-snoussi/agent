# Phases 2-5 Implementation Notes

Summary of work completed after the Phase 1 gap analysis to satisfy the remaining QBRIX requirements.

## Phase 2 – Color Pipeline & Image Quality
- Replaced the monolithic preprocessing logic with `ImagePreprocessor`, adding smart exposure control, bilateral denoise, optional face-aware saliency cropping, and grid-aligned resampling that honors the 10k-cell constraint.
- Refactored palette quantization in `grid_index_map.py` to accept preprocessed Lab data, cache palette conversions, apply luminance-aware tie breaking, and expose configurable spatial smoothing and POPART edge bias through `config.yaml`.
- Recorded preprocessing metadata (crop rectangle, face boxes, saliency center, adjustments) so every run is auditable and downstream previews share the same grid index map.

## Phase 3 – QBRIX Assembly Layout
- Rebuilt `src/diamondkit/pdf.py` to match the reference instruction booklets: fixed headers, A4 landscape margins, numeric grids anchored to crop marks, overlap shading on pattern pages, and mini-maps that highlight tile positions plus seam overlaps.
- Tightened typography (Helvetica bold digits sized to meet >=1.2 mm x-height), added registration crosses, tile coordinate callouts, and instruction copy that explains overlaps, printing rules, and navigation.

## Phase 4 – Output Contract & Metadata
- Consolidated metadata writing via `_write_metadata_bundle`, ensuring both `kit_metadata.json` and the legacy `metadata.json` include tiling maps, grid hash, fixed palette declarations, preprocessing notes, warnings, and output manifests.
- Harmonized previews and style comparison renders so every derivative asset references the exact same grid and palette order (no hidden re-quantization).
- Updated CSV/JSON/PDF consumers (demo scripts, CLI, web app) to rely on the new field names, scale-factor data, and warning surfaces.

## Phase 5 – Validation & Tests
- Added `tests/test_qbrix_pipeline.py` covering metadata contracts, palette invariants, and inventory ordering. Tests run via `pytest -q tests/test_qbrix_pipeline.py`.
- Wired scale-factor, overlap, and rare-color checks into `quality_gates.py`, surfacing actionable warnings plus auto-fix suggestions in the CLI/demo outputs.
- Documented acceptance criteria in this file so future phases can trace which modules satisfy each requirement.
