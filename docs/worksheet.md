# SCynergy 2026 - GeoAI Workshop — Worksheet 

## 📘 Notebook 1 — Understanding Data Acquisition

1. What problem does STAC solve in geospatial workflows?  
   → Think about how data is discovered and filtered.

2. Why do we define multiple temporal phases (pre-event, event, post-event)?  
   → What additional information do they provide?

3. What factors determine the “best” satellite scene?  
   → Consider differences between optical and radar data.

---

## 📗 Notebook 2 — Understanding Data Packaging

1. Why must all datasets be aligned onto a common grid?  
   → What would happen if they were not?

2. What is reprojection, and why is it necessary?  
   → Think about coordinate systems and pixel alignment.

3. Why do we split large images into smaller chips?  
   → Consider GPU limitations and model requirements.

4. How does the workflow handle missing or invalid data?  
   → Why is this important for AI models?

---

## 📕 Notebook 3 — Understanding Model Inference

1. What does each dimension of the input tensor represent?  
   → [Batch, Channels, Time, Height, Width]

2. Why is DEM repeated across the time dimension?  
   → Does the terrain change over time?

3. What does the model output represent?  
   → How would you interpret the prediction?

---

## Reflection Questions

* Which part of the pipeline was most surprising or new to you?
* Where do you think errors or uncertainties could arise?
* How could this workflow be adapted to another application (e.g., wildfire, agriculture)?
