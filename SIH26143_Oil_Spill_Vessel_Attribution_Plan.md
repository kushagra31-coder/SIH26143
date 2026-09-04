# SIH 26143 --- Oil Spill Detection & Vessel Attribution

## 1. Problem in one line

Build an automated pipeline that detects oil slicks from satellite
imagery, estimates where/when the spill likely originated using
oceanographic and meteorological data, reconstructs historical AIS
vessel traffic around that origin window, and ranks candidate vessels
using explainable spatial, temporal, trajectory, and behavioural
evidence.

## 2. What the SIH problem requires

1.  Detect and characterise oil spills from SAR/EO imagery.
2.  Calculate slick geometry such as area, centroid and bounding region;
    estimate age if feasible.
3.  Use oceanographic and meteorological data to model slick drift.
4.  Hindcast the slick toward a probable origin location/time.
5.  Forecast forward movement.
6.  Reconstruct historic AIS traffic around the origin window.
7.  Filter irrelevant vessels.
8.  Score candidate vessels using proximity, trajectory, temporal
    correlation and behavioural anomalies.
9.  Provide a visual interface.

Official/community-archived PS reference:
https://sih2026.vuce.in/ps/SIH26143

## 3. Proposed end-to-end architecture

``` text
Sentinel-1 SAR
      ↓
SAR preprocessing
      ↓
Oil-spill segmentation
      ↓
Slick geometry + confidence
      ↓
Wind + ocean-current data
      ↓
OpenDrift / OpenOil
      ↓
Backward drift / hindcast
      ↓
Probable origin region + time window
      ↓
Historical AIS
      ↓
Candidate filtering
      ↓
Spatial + temporal + trajectory + AIS-behaviour features
      ↓
Evidence-fusion / attribution score
      ↓
Ranked candidate vessels
      ↓
GIS dashboard + report
```

## 4. What is already available

### Oil-spill segmentation

**Verified:** `Harsha0112/Oil-Spill-Detection`

https://github.com/Harsha0112/Oil-Spill-Detection

The repository contains U-Net and DeepLabV3 notebooks and describes
Sentinel-1 SAR oil-spill detection plus AIS integration. It references
the `nabilsherif/oil-spill` Kaggle dataset.

Use it as a **baseline/reference**, not as the complete SIH solution.

### Oil drift

**Verified:** `OpenDrift/OpenDrift`

https://github.com/OpenDrift/opendrift

OpenDrift is an open-source ocean trajectory framework and contains the
`OpenOil` model for oil trajectory/fate modelling.

OpenOil source:

https://github.com/OpenDrift/opendrift/blob/master/opendrift/models/openoil/openoil.py

OpenDrift examples also show seeding oil from a satellite-detected
contour.

### AIS anomaly detection

**Verified:** `sansastra/Anomaly-Detection`

https://github.com/sansastra/Anomaly-Detection

Useful for AIS on/off switching, unusual turns and trajectory
deviations.

**Verified:** `CIA-Oceanix/GeoTrackNet`

https://github.com/CIA-Oceanix/GeoTrackNet

Useful reference for maritime AIS trajectory representation and anomaly
detection. It includes MarineC-related preprocessing/data references.

**Verified:** `LeoPits/Vessels-anomaly-detection-with-AIS-data`

https://github.com/LeoPits/Vessels-anomaly-detection-with-AIS-data

Useful reference for waypoint extraction, DBSCAN-based route learning
and normal-route modelling.

### Repositories that should NOT be treated as verified

`nobleaustine/OSDetector` --- exact repository not independently
verified.

`sahilbagde22/oil-spill-detection` --- exact repository not
independently verified.

Do not make either a project dependency until the exact repository is
confirmed.

## 5. Data sources

### Sentinel-1

Copernicus Data Space:

https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-1

Documentation:

https://documentation.dataspace.copernicus.eu/Data/Sentinel1.html

Sentinel-1 products are made available free of charge. SAR is useful for
marine observation because it can operate day/night and is not dependent
on optical cloud visibility.

### Oil-spill training data

Reference used by the verified Harsha repository:

https://www.kaggle.com/datasets/nabilsherif/oil-spill

Additional dataset reference:

https://github.com/SARDEEP1/OSLM

Verify current availability, licensing, labels and geographic coverage
before making a dataset a hard dependency.

### Historical AIS

MarineCadastre:

https://marinecadastre.gov/ais/

AccessAIS:

https://marinecadastre.gov/accessais/

MarineCadastre provides historical AIS data, but the current AccessAIS
page says its ordering service is unavailable while bulk downloads
remain available. Verify the exact date range and geographic coverage
before development.

MarineCadastre AIS GitHub:

https://github.com/ocm-marinecadastre/ais-vessel-traffic

## 6. Our unique contribution

Do NOT pitch:

> "We use AI to detect oil spills."

That is already an established research/product area.

The differentiated layer should be:

``` text
SAR detection
      +
Slick geometry
      +
Physical drift hindcasting
      +
Origin uncertainty
      +
Historical AIS reconstruction
      +
AIS gap detection
      +
Normal-route deviation
      +
Spatial/temporal alignment
      ↓
Explainable evidence fusion
      ↓
Ranked candidate vessels
```

### Core contribution

**Uncertainty-aware, evidence-weighted vessel attribution.**

Instead of producing:

``` text
Vessel A = 87% culprit
```

produce:

``` text
Vessel A — Rank #1

Spatial correlation:       0.91
Temporal correlation:      0.87
Trajectory consistency:    0.82
AIS-gap evidence:          0.70
Route deviation:           0.65

Overall evidence score:    0.83

Status:
Strong candidate — requires human investigation
```

This avoids the scientifically weak claim that an algorithm has "proved"
the culprit.

## 7. Proposed attribution features

### Spatial

Distance between vessel trajectory and inferred origin region.

### Temporal

Was the vessel present during the estimated release window?

### Trajectory

Does the vessel's movement make sense relative to the inferred origin
and drift path?

### AIS gap

Did the vessel have a significant AIS transmission gap around the
relevant time?

Important: an AIS gap is **supporting evidence, not proof of illegal
discharge**.

### Route deviation

Does the vessel deviate from its historically learned normal route?

## 8. Uncertainty layer

Carry uncertainty through the system.

Bad:

``` text
Origin = exact coordinate
Culprit = Vessel A
```

Better:

``` text
Oil detection confidence: 0.87

Probable origin:
polygon / probability region

Origin time:
08:20–09:10 UTC

Candidate ranking:
1. Vessel A — strong evidence
2. Vessel B — moderate evidence
3. Vessel C — weak evidence
```

OpenDrift reverse modelling should be treated as an estimate, not an
exact reconstruction.

## 9. False-positive handling

A dark SAR region is not automatically oil.

Potential look-alikes include:

-   low-wind areas
-   biogenic films
-   rain effects
-   other SAR artefacts
-   calm-water regions

If feasible, use a multi-class or look-alike-aware model:

``` text
Sea
Oil
Look-alike
Ship
Land
```

A newer verified public reference is:

https://github.com/m7mdehab/oil-spill-detection

Its README describes five-class Sentinel-1 SAR segmentation and
emphasizes oil-specific IoU/recall rather than relying only on overall
pixel accuracy.

## 10. Historical-case strategy

Do NOT start with a live global system.

Choose one documented historical spill:

``` text
Known spill
   ↓
SAR scene
   ↓
Detection
   ↓
Drift hindcast
   ↓
AIS reconstruction
   ↓
Candidate ranking
```

Only add a second case after the complete pipeline works.

This matches the project deep-dive recommendation in the team's SIH
analysis: start with one historical case, then generalize if time
remains.

## 11. 36-hour build order

### 0--3 h --- Data and environment

-   Pick historical spill case.
-   Obtain SAR.
-   Obtain AIS.
-   Verify timestamps.
-   Install OpenDrift.
-   Verify wind/current data.

**Hard gate:** if the selected case has unusable AIS, change the case
immediately.

### 3--9 h --- Detection

Get a baseline segmentation model running.

Do not spend the majority of the hackathon optimizing the detector
before the complete pipeline exists.

### 9--14 h --- Slick characterisation

Implement:

-   mask cleanup
-   polygon extraction
-   area
-   centroid
-   bounding box
-   confidence
-   metadata

### 14--20 h --- Drift

Connect:

``` text
Slick
 ↓
OpenOil
 ↓
Wind/current
 ↓
Backward simulation
```

Generate probable origin region/time.

### 20--26 h --- AIS

Implement:

``` text
Origin region + time
        ↓
AIS filtering
        ↓
Candidate vessels
        ↓
Trajectory reconstruction
```

### 26--30 h --- Attribution

Add:

-   proximity score
-   temporal score
-   trajectory score
-   AIS-gap evidence
-   route-deviation evidence

### 30--34 h --- Dashboard

Show:

-   slick
-   origin region
-   backward trajectory
-   forward trajectory
-   vessels
-   candidate ranking
-   evidence breakdown

### 34--36 h --- Demo and validation

Prepare:

-   one complete case
-   metrics
-   architecture
-   limitations
-   source acknowledgements
-   judge Q&A

## 12. Evaluation

### Detection

Report:

-   precision
-   recall
-   F1
-   IoU
-   Dice

Focus on oil-class metrics.

### Drift

Where ground truth exists, compare predicted drift/origin against later
observations or documented trajectories.

Possible metrics:

-   centroid error
-   distance error
-   trajectory overlap

### Attribution

Do not claim attribution accuracy without ground truth.

Evaluate:

-   candidate reduction
-   ranking quality on documented cases
-   evidence consistency
-   explainability

## 13. What NOT to build

-   Live global monitoring in the first version.
-   A custom ocean-physics engine.
-   A huge custom AIS neural network.
-   A black-box "culprit probability".
-   A dashboard before the core pipeline works.

## 14. Recommended technology stack

### ML

-   Python
-   PyTorch or TensorFlow
-   U-Net / SegFormer / DeepLabV3+ baseline

### Geospatial

-   Rasterio
-   GDAL
-   GeoPandas
-   Shapely
-   pyproj

### AIS

-   Pandas
-   GeoPandas
-   NumPy
-   scikit-learn

### Drift

-   OpenDrift
-   OpenOil

### Backend

-   FastAPI

### Dashboard

Fastest MVP:

-   Streamlit
-   Folium

Alternative:

-   React
-   MapLibre / Leaflet

## 15. Repository/license caution

OpenDrift is GPL-2.0 licensed:

https://github.com/OpenDrift/opendrift/blob/master/LICENSE

`sansastra/Anomaly-Detection` is MIT licensed.

GeoTrackNet contains explicit license files in its repository.

Before copying code:

1.  Check the repository license.
2.  Preserve required attribution.
3.  Record repository/version used.
4.  Keep third-party code distinguishable from your original work.
5.  Do not claim an existing algorithm as your original invention.

The Harsha repository does not present a clear license in the
information verified here; treat it as reference material until its
licensing is confirmed.

## 16. Final pitch

> **We combine satellite SAR oil-slick detection with physical drift
> hindcasting and historical AIS reconstruction to reduce a large
> maritime traffic search space to a small, explainable set of candidate
> vessels. Each candidate is ranked using independent spatial, temporal,
> trajectory and behavioural evidence, while uncertainty is explicitly
> preserved for human investigation.**

## 17. One-line differentiator

**Detect the spill → reconstruct where it came from → find who was there
→ explain why they are candidates.**

## 18. Source verification notes

Verified during preparation:

-   SIH26143 problem description: https://sih2026.vuce.in/ps/SIH26143
-   Harsha oil-spill repository:
    https://github.com/Harsha0112/Oil-Spill-Detection
-   OpenDrift: https://github.com/OpenDrift/opendrift
-   OpenOil source:
    https://github.com/OpenDrift/opendrift/blob/master/opendrift/models/openoil/openoil.py
-   GeoTrackNet: https://github.com/CIA-Oceanix/GeoTrackNet
-   AIS anomaly reference:
    https://github.com/sansastra/Anomaly-Detection
-   AIS route reference:
    https://github.com/LeoPits/Vessels-anomaly-detection-with-AIS-data
-   Sentinel-1:
    https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-1
-   MarineCadastre AIS: https://marinecadastre.gov/ais/
-   MarineCadastre AIS GitHub:
    https://github.com/ocm-marinecadastre/ais-vessel-traffic
-   OSLM dataset reference: https://github.com/SARDEEP1/OSLM
-   Additional oil-spill pipeline reference:
    https://github.com/m7mdehab/oil-spill-detection
