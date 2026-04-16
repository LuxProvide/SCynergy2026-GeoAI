# SCynergy 2026 - GeoAI Workshop — Cheat Sheet 

## 🧭 End-to-End Pipeline 
This workshop demonstrates a full GeoAI workflow:

1. **Data Acquisition** – selecting relevant satellite scenes using STAC
2. **Data Alignment** – ensuring all datasets share the same spatial grid
3. **Data Packaging** – converting large scenes into model-ready tensors
4. **Model Inference** – applying a pretrained TerraMind flood model

👉 Key message: *AI performance depends heavily on correct geospatial preprocessing.*

---

## 📘 Notebook 1 — Data Acquisition

### Key Concept: STAC
STAC (SpatioTemporal Asset Catalog) is a structured way to search geospatial datasets by location, time, and metadata. It means that, with STAC, you are **querying a catalog**, not manually downloading files.

### Temporal Phases
Why the workflow uses:

* pre-event
* event
* post-event

👉 Floods are dynamic — comparing time periods improves detection.

### Scene Selection Logic
Why:

* Sentinel-2 → filter clouds
* Sentinel-1 → prioritize time + coverage

👉 “Best scene” depends on task, not just recency.

### Output: Manifest
This ensures reproducibility by recording selected scenes.

---

## 📗 Notebook 2 — Data Packaging 

### Common Grid
All datasets are transformed into:

* same coordinate system
* same resolution
* same spatial extent

👉 This enables pixel-wise comparison across modalities.

### Reprojection
Reprojection is a geometric transformation, not just format conversion.

### Handling Missing Data
Invalid pixels become NaN and are masked to prevent incorrect learning.

### Chip Extraction
Large images are divided into smaller patches (e.g., 256×256):

* fits GPU memory
* enables batch processing

### Memory Efficiency
Data is stored using memory mapping to handle large files efficiently.

---

## 📕 Notebook 3 — Model Inference

### Tensor Structure
Inputs follow shape:
[B, C, T, H, W] meaning [Batch, Channels, Time, Height, Width]

### Multimodal Fusion
The model combines:

* optical (Sentinel-2)
* radar (Sentinel-1)
* terrain (DEM)

### Temporal Handling
DEM is repeated across time because the model expects temporal inputs.

### Output
Model predicts flood vs non-flood segmentation masks.

---

## Important Notes
* Alignment enables fusion.
* Data quality drives model quality.
* This pipeline is reusable beyond floods.
