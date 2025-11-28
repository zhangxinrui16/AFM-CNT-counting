# AFM-CNT-counting

A Streamlit app that estimates carbon nanotube (CNT) density from AFM height images. The tool enhances line-like structures, extracts skeletons, merges split segments, and handles X/Y junctions to avoid double-counting.

## Features
- Frangi-based ridge enhancement to highlight CNTs while suppressing point impurities.
- Adaptive thresholding and morphological cleanup to remove small bright artifacts.
- Skeletonization with gap bridging for long CNTs that span disconnected regions.
- Junction-aware counting: merges nearly colinear branches (Y-type) while preserving distinct crossings (X-type).
- Interactive UI: upload image, enter scan dimensions (μm), view overlays, counts, and density.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
streamlit run app.py
```
If you accidentally run `python app.py`, the script now delegates to `streamlit run` automatically to avoid the missing `ScriptRunContext` warning.
1. Upload an AFM height-channel image (png/jpg/tiff).
2. Enter scan length (μm) and optional width (defaults to length).
3. View detected CNT overlays and computed density (tubes per μm²).

## Notes
- Default thresholds target images similar to the sample provided; adjust parameters in `app.py` if your images differ significantly.
- Density is computed as detected tube roots divided by the scanned area.
