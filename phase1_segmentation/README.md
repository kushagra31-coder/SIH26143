SIH26143 — Oil Spill Detection & Vessel Attribution
Overview
SIH26143 is an SIH MVP for oil-spill candidate detection and vessel attribution using satellite SAR imagery and historical AIS data.
The intended pipeline is:
```text
Sentinel-1 SAR
    ↓
SAR preprocessing
    ↓
U-Net segmentation
    ↓
spill probability / binary mask
    ↓
geospatial candidate regions
    ↓
historical AIS filtering
    ↓
candidate-vessel ranking
```
The system is decision support. It identifies candidate spill regions and ranks vessels using spatial, temporal, and trajectory evidence. It does not establish that a vessel caused the spill.
---
Case Study
MV Wakashio — Mauritius, 2020
Current real-scene analysis uses:
```text
Satellite:    Sentinel-1B
Date:         2020-08-10
Polarization: VV

Product:
S1B_IW_GRDH_1SDV_20200810T013755_20200810T013820_022854_02B625_672D.SAFE
```
Focused AOI:
```text
Longitude: 57.65 → 57.90
Latitude:  -20.55 → -20.30
```
---
Repository Structure
```text
SIH26143/
├── phase0_validation/
├── phase1_segmentation/
│   ├── training/
│   ├── preprocessing/
│   ├── inference/
│   ├── postprocessing/
│   ├── evaluation/
│   └── dev/
├── phase2_drift/
├── phase3_ais/
├── phase4_matching/
├── phase5_output/
├── .gitignore
└── README.md
```
---
Phase 1 — SAR Oil-Spill Segmentation
Training Dataset
Kaggle SOS SAR Oil Spill Segmentation Dataset:
```text
bitsandlayers/sar-oil-spill-segmentation-dataset-sos
```
Only the Sentinel-1 subset is used.
Prepared split:
```text
3154 training images
200 validation images
```
Training images are `256×256×3` uint8 images. The three channels were empirically found to be identical, so the input is effectively grayscale SAR repeated across three channels.
TensorFlow preprocessing normalizes image values to approximately `[0, 1]`.
Model
```text
Architecture: U-Net
Base filters: 32
Input:        256×256×3
Output:       256×256×1
Optimizer:    Adam
Batch size:   2
```
Loss experiments included:
```text
BCE + Dice
Tversky-based losses
BCE + Dice + Tversky
```
Model optimization was stopped once the MVP pipeline was sufficiently functional.
Inference Threshold
The current locked threshold is:
```text
0.20
```
This was selected through validation-set threshold analysis.
---
Real Sentinel-1 Preprocessing
The real Wakashio scene was processed in ESA SNAP.
```text
Read
 ↓
Apply Orbit File
 ↓
Thermal Noise Removal
 ↓
Calibration
 ↓
Terrain Correction
 ↓
GeoTIFF
```
Thermal Noise Removal
```text
Polarization: VV
Remove noise: ON
Reintroduce noise: OFF
Output noise: OFF
Clip negatives: ON
```
Calibration
```text
Polarization: VV
Product: Sigma0
Output: linear
```
Terrain Correction
```text
Input: Sigma0_VV
DEM: SRTM 1Sec HGT
DEM resampling: bilinear
Image resampling: bilinear
Target resolution: 10 m
CRS: WGS84 / EPSG:4326
Radiometric normalization: OFF
```
Processed output:
```text
satellite data/processed/wakashio_sigma0_vv_tc.tif
```
Approximate properties:
```text
30590 × 22880
1 band
float32
EPSG:4326
~10 m pixel spacing
```
---
Model-Compatible SAR Representation
The processed Sigma0 raster was converted to dB and mapped to 8-bit intensity.
Two ranges were tested:
```text
DB25: -25 dB → 0, 0 dB → 255
DB30: -30 dB → 0, 0 dB → 255
```
DB25 caused severe overprediction.
DB30 is the locked working representation.
```text
-30 dB → 0
  0 dB → 255
```
The resulting single-band grayscale representation is repeated across 3 channels to match the training-image format.
---
Tiling and Stitching
Tile size:
```text
256 × 256
```
Stride:
```text
192 pixels
```
Overlap:
```text
64 pixels
```
Edge padding:
```text
reflect
```
Total tiles:
```text
225
```
The overlapping tile predictions are stitched into a georeferenced full-scene probability map.
A neighbor overlap check between `tile_00109` and `tile_00110` gave:
```text
Correlation ≈ 0.9846
IoU @ 0.20 ≈ 0.9114
```
indicating good spatial consistency at tile boundaries.
---
Post-Processing
```text
Probability map
    ↓
land / ocean filtering
    ↓
threshold 0.20
    ↓
8-connected components
    ↓
remove components < ~100 pixels
    ↓
polygonization
    ↓
area / centroid / bbox / confidence
```
Natural Earth 10 m land data were used for land masking.
Land masking reduces obvious land contamination but does not eliminate all coastal SAR look-alikes.
---
Final Prediction Outputs
Main outputs:
```text
wakashio_probability_db30.tif
wakashio_binary_db30.tif
```
Binary mask semantics:
```text
0   = background
255 = predicted spill candidate
```
Candidate regions are also exported as GIS polygons, preferably GeoJSON/GeoPackage.
These are model-derived candidate regions, not confirmed oil.
---
UNOSAT Reference
Independent reference source:
UNITAR–UNOSAT
Product:
> Satellite detected potential oil extent as of 10 August 2020 in Pointe d'Esny Reef, Republic of Mauritius
Product ID:
```text
2885
```
Selected reference:
```text
TSX_20200810_OilSpillExtent_ReefPointeEsny.shp
```
The reference is based on TerraSAR-X imagery acquired on 10 August 2020.
Important limitation
UNOSAT is a credible independent satellite-analysis source, but the product was preliminary and not field-validated. Radar-based oil-spill interpretation can also contain look-alike effects.
Therefore this repository describes it as a:
```text
UNOSAT satellite-derived reference spill extent
```
not absolute field ground truth.
---
Real-Scene Validation
Prediction:
```text
wakashio_binary_db30.tif
```
Reference:
```text
TSX_20200810_OilSpillExtent_ReefPointeEsny.shp
```
Corrected confusion matrix:
```text
TP = 122,049
TN = 7,447,034
FP = 154,522
FN = 21,484
```
Core metrics:
Metric	Result
IoU / Jaccard	40.95%
Dice / F1	58.10%
Precision	44.13%
Recall / Sensitivity	85.03%
Specificity	97.97%
Accuracy	97.73%
Balanced Accuracy	91.50%
MCC	0.6032
Cohen's Kappa	0.5706
Area comparison:
```text
Predicted area:     ≈25.81 km²
UNOSAT reference:   ≈13.39 km²
Intersection:       ≈11.39 km²
Union:              ≈27.81 km²
Predicted/reference ≈1.93×
```
Interpretation
The model has:
high recall;
moderate spatial agreement;
relatively low precision;
substantial overprediction;
persistent coastal/boundary false positives.
For a Round 1 MVP, these results are sufficient to demonstrate an end-to-end candidate-detection pipeline, but should not be presented as production-grade or field-validated segmentation accuracy.
---
## Interpretation

The model has:

- **High recall**
- **Moderate spatial agreement**
- **Relatively low precision**
- **Substantial overprediction**
- **Persistent coastal/boundary false positives**

For a Round 1 MVP, these results are sufficient to demonstrate an end-to-end candidate-detection pipeline, but should not be presented as production-grade or field-validated segmentation accuracy.

---

## Important Limitations

### 1. SAR Look-Alikes

Dark SAR regions can come from phenomena other than oil, including calm water, wind effects, geometry, and coastal structures.

### 2. Domain Gap

The SOS training data and the real Wakashio scene differ in geography and acquisition conditions, so transfer introduces domain shift.

### 3. Reference Uncertainty

The UNOSAT reference is satellite-derived and preliminary rather than field-validated.

### 4. Attribution Uncertainty

A vessel being close to a detected spill does not prove that it caused the spill.
---
AIS Integration
The satellite module should provide:
```text
spill polygon
spill centroid
spill bounding box
spill area
image date/time
prediction confidence
```
Conceptual interface:
```text
spill_features.json
       ↓
AIS search region
       ↓
historical AIS
       ↓
spatial + temporal filtering
       ↓
candidate tracks
       ↓
proximity + temporal + trajectory scoring
       ↓
ranked_vessels.json
```
Useful AIS fields:
```text
MMSI
timestamp
latitude
longitude
SOG
COG
vessel type
IMO (if available)
```
The vessel score is an evidence/ranking score, not a culpability score.
---
Data Policy
Large/raw datasets and generated satellite products are intentionally kept outside Git.
Do not commit:
```text
*.SAFE
*.tif
*.tiff
*.jp2
training datasets
large prediction outputs
model checkpoints
```
The repository should primarily contain:
```text
source code
configuration
documentation
small metadata
evaluation methodology/results
```
---
Key Files
Model:
```text
phase1_segmentation/checkpoints/best_model.keras
```
Real-scene processing:
```text
satellite data/processed/wakashio_sigma0_vv_tc.tif
satellite data/model_candidates/wakashio_candidate_db_-30_to_0.tif
satellite data/model_input_db30/wakashio_aoi_candidate_db30.tif
```
Prediction:
```text
satellite data/predictions_full_db30/wakashio_probability_db30.tif
satellite data/predictions_full_db30/wakashio_binary_db30.tif
```
UNOSAT reference:
```text
UNOSAT/TSX_20200810_OilSpillExtent_ReefPointeEsny.shp
```
---
## Key Sources

### UNOSAT

UNITAR–UNOSAT — *Satellite detected potential oil extent as of 10 August 2020 in Pointe d'Esny Reef, Republic of Mauritius*, Product 2885.

### Wakashio Remote Sensing

Rajendran et al. (2021) — *Detection of Wakashio oil spill off Mauritius using Sentinel-1 and 2 data: Capability of sensors, image transformation methods and mapping.*

**Environmental Pollution**, 274, 116618.

DOI: `10.1016/j.envpol.2021.116618`

### Wakashio Baseline Assessment

Rajendran et al. (2022) — *History of a disaster: A baseline assessment of the Wakashio oil spill on the coast of Mauritius, Indian Ocean.*

**Marine Pollution Bulletin**, 175, 113330.

### Training Dataset

SAR Oil Spill Segmentation Dataset (SOS)

Kaggle: `bitsandlayers/sar-oil-spill-segmentation-dataset-sos`

### Software & Geospatial Data

- **ESA SNAP** — Sentinel-1 preprocessing
- **QGIS** — visualization and geospatial analysis
- **Natural Earth 10m** — land masking
- **TensorFlow / Keras** — U-Net training and inference
- **GeoPandas / Rasterio / NumPy** — geospatial processing
---
Current Status
```text
✅ Phase 0 validation
✅ Phase 1 U-Net segmentation
✅ Real Wakashio Sentinel-1 preprocessing
✅ Real-scene inference
✅ Georeferenced candidate extraction
✅ UNOSAT reference validation

```
---
Final MVP Description
> **SIH26143 applies a U-Net trained on Sentinel-1 oil-spill segmentation data to a real Sentinel-1B scene of the 2020 MV Wakashio spill, performs SAR preprocessing and DB30 intensity conversion, extracts georeferenced candidate spill regions, validates them against an independent UNOSAT satellite-derived reference extent, and passes spatial/temporal spill information to a historical AIS module for candidate-vessel ranking without claiming vessel responsibility.**
