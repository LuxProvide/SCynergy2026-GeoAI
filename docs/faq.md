# SCynergy 2026 - GeoAI Workshop — FAQ 

## ❓ Why not use a single satellite image?
Floods are dynamic events. Using multiple timestamps and modalities allows the model to detect changes and improve robustness.

---

## ❓ Why is reprojection necessary?
Different datasets come in different coordinate systems and resolutions.  
Reprojection ensures that all pixels align spatially so the model can combine them correctly.

---

## ❓ Why split images into smaller patches (chips)?
Satellite scenes are very large.  
Models require fixed-size inputs, and GPUs have memory limits.  
Chipping allows scalable processing.

---

## ❓ Are we training a model in this workshop?
No.  
We are using a **pretrained TerraMind model** to perform inference.  
The training process is demonstrated separately in the IBM tutorial.

---

## ❓ Why include DEM (elevation data)?
Elevation helps constrain flood predictions:
- water flows downhill
- low-lying areas are more likely to flood

---

## ❓ What are the main limitations of this workflow?

- **Cloud cover (Sentinel-2)** → missing optical data  
- **SAR noise (Sentinel-1)** → harder to interpret  
- **Resolution differences** → potential misalignment  
- **Temporal gaps** → missing key moments  

---

## ❓ What is the most important takeaway?
Accurate AI predictions depend heavily on **correct geospatial preprocessing and alignment**, not just the model itself.
