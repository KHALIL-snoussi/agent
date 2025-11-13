# Phase 1 - QBRIX Gap Analysis

## Implementation Map
- **Image preprocessing** lives in `src/diamondkit/kit_generator.py:52-155` (straight PIL load plus optional center crop) with a secondary neutral adjustment inside `src/diamondkit/grid_index_map.py:198-254`; there is no dedicated preprocessing module.
- **Palette quantization & grid map** happen in `src/diamondkit/grid_index_map.py:70-330` (ColorQuantizerFixed + create_grid_index_map) with hard-coded smoothing and no style-specific edge handling.
- **Quality metrics** are computed in `src/diamondkit/quality_assessor.py:28-235` and `assess_quality` is called from `src/diamondkit/kit_generator.py:92-115`; quality gates live in `src/diamondkit/quality_gates.py`.
- **PDF layout** (title, legend, instructions, pattern tiles) is defined in `src/diamondkit/pdf.py`.
- **Metadata/exports** are assembled in `src/diamondkit/kit_generator.py:450-707`, while previews/CSV/JSON are emitted from `_generate_all_outputs` in the same file.

## Color & Quantization Gaps
- `_load_and_validate_image` and `_apply_smart_cropping` in `src/diamondkit/kit_generator.py:156-220` only do center crops and entropy warnings. There is no content-aware crop, face detection, saliency map, or recorded crop rectangle, so every auto-crop ignores the "interesting" region requirement.
- `crop_rect_norm` is always `None` in `_generate_json_metadata` (`src/diamondkit/kit_generator.py:509-535`), so metadata never records what portion of the source image was used.
- Preprocessing is limited to mild histogram stretching and sharpening after the image is already resized down to the grid (`src/diamondkit/grid_index_map.py:198-254`). There is no working-resolution pipeline (800-1200 px long side), no white-balance control beyond a crude global clamp, no gamma option, and no pre-quantization denoise before decimation.
- POPART edge protection only happens inside the style preview helper (`src/diamondkit/kit_generator.py:403-445`). The actual quantizer never biases strong edges toward DMC 310, so outlines disappear even when the preview shows them.
- `create_grid_index_map` is always called with `enable_smoothing=True` (`src/diamondkit/kit_generator.py:79`), but the kernel size/strength is not configurable via `config.yaml`, so there is no way to tune or disable the 3x3 majority filter per requirement.
- Cluster tie-breaking in `ColorQuantizerFixed._quantize_pixels_deltae` (`src/diamondkit/grid_index_map.py:248-331`) only penalises the lightest swatch; there is no luminance-weighted fallback to keep mid-tone detail when DeltaE values tie, which hurts portraits.
- Style comparison previews just recolor the already-quantized RGB using filters (`src/diamondkit/kit_generator.py:320-450`). They do **not** remap cluster ids to the other fixed palettes, so `preview_vintage`/`preview_popart` do not reflect what the grid would look like in those palettes.

## Layout & PDF Gaps vs QBRIX
- Pattern grids are centered with `start_x = (self.page_w - tile_width)/2` (`src/diamondkit/pdf.py:584`), ignoring the 12 mm margins and the crop marks drawn at the page edges (`src/diamondkit/pdf.py:534-570`). The numeric matrix therefore floats inside the sheet instead of aligning to the cut lines like the reference QBRIX instructions.
- Row labels use `start_y` instead of `start_x` when placing the left-side indices (`src/diamondkit/pdf.py:661`), so row numbers are rendered far from the grid.
- Symbol font size is fixed at `cell_size_pt * 0.4` (`src/diamondkit/pdf.py:48`), which yields roughly 1.1 mm x-height at 2.8 mm cells  below the >=1.2 mm legibility requirement even though quality gates assume it passes.
- The title page pulls warnings from `metadata.get('warnings', [])` (`src/diamondkit/pdf.py:287`), but the metadata bundle never defines this key, so quality warnings (SSIM < 0.75, DeltaE > 12, rare colors) never surface.
- Quality status badges on the title page still treat DeltaE_mean < 40 and DeltaE_max < 60 as "good" (`src/diamondkit/pdf.py:256-278`), which conflicts with the stricter DeltaE_max <= 12 gate documented in `quality_assessor`.
- Pattern headers do not announce the style/palette or overlap instructions; overlaps are neither tinted nor indicated on the mini-map, so users cannot see the 2-cell shared seam the reference sheet highlights.
- Assembly instructions (`src/diamondkit/pdf.py:389-452`) are generic text and do not mention mini-map usage, overlap alignment, or coordinate conventions seen in instruction2.pdf.

## Metadata & Output Contract Issues
- The JSON written by `_generate_json_metadata` contains `tiling_map`, `color_usage`, and `crop_rect_norm`, but the `kit_metadata.json` emitted later from `_compile_comprehensive_metadata` (`src/diamondkit/kit_generator.py:618-704`) omits those fields. The contract in the brief specifically requires them in `kit_metadata.json`.
- `kit_metadata.json` lacks an explicit `quality_warnings`/`quality_risks` top-level struct; they are nested under `quality_assessment` but never bubbled up, which is why the PDF cannot show them.
- `metadata.get('ssim')` on the title page actually points to the string from `_generate_json_metadata`, not the final SSIM stored inside `quality_assessment`, so the PDF can drift from the saved metrics.
- There are two metadata files (`metadata.json` and `kit_metadata.json`) with different schemas in the output folder, causing downstream tools to guess which one is authoritative.

## Testing & Validation Gaps
- The `tests/` directory is empty  there are no automated checks for the fixed 7-color palette, <=10,000-cell enforcement, tiling coverage, or quality-gate warnings that the acceptance criteria call for.

These gaps explain the current delta between the repo's output and the provided QBRIX reference: color fidelity suffers because preprocessing/quantization ignores content and hard edges, and the PDF lacks the structural cues (alignment, indices, overlap hints, warning summaries) that make the instruction sheets usable.
