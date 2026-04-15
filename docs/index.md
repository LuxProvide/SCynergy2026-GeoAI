# SCynergy 2026 - GeoAI Workshop — Overview

Author: Eun-Kyeong Kim (eun-kyeong.kim@lxp.lu), LuxProvide S.A

## What is this workshop about?

This workshop introduces how **multimodal geospatial data** can be used together with a **foundation model (TerraMind)** to perform flood mapping. This tutorial is given in the context of the [Scynergy 2026 event](https://www.scynergy.events/) and [EPICURE](https://epicure-hpc.eu/).

You will learn how satellite data from different sources (Sentinel-1, Sentinel-2, and DEM) can be:
- discovered
- aligned
- transformed
- and used for AI-based prediction

---

## Learning Objectives

By the end of this workshop, you will:

- Understand what **GeoAI** and **geospatial foundation models (GeoFMs)** are
- Learn why **multimodal data** is important for Earth observation tasks
- Understand how geospatial data must be **aligned and prepared** before AI can be applied
- Run a **TerraMind flood model** on real satellite data
- Interpret model predictions and their limitations

---

## Workshop Structure

### Part 1 — Concepts (20 minutes)
- GeoAI and multimodal Earth observation
- TerraMind foundation model
- Geospatial data concepts (raster, reprojection, alignment)
- End-to-end workflow overview

### Part 2 — Hands-on (60 minutes)
You will work through three notebooks:

1. **Data Acquisition**
   - Search and select satellite data using STAC

2. **Data Packaging**
   - Align and transform data into model-ready format

3. **Model Inference**
   - Run TerraMind to detect flooded areas

---

## Key Idea

👉 *AI models are only as good as the data they receive.*

A major focus of this workshop is understanding how **geospatial preprocessing enables multimodal AI**.

---

## Platform

This workshop runs on:
- **MeluXina GPU cluster**
- **Open OnDemand JupyterLab**

No prior HPC experience is required.

---

## Who is this for?

This workshop is designed for:
- AI / ML practitioners interested in geospatial data
- Remote sensing / GIS experts exploring AI methods
- Researchers working with Earth observation data

---

## What you will build

By the end, you will have:
- Executed a complete **GeoAI pipeline**
- Understood how TerraMind works in practice
- Gained a reusable workflow for other EO applications

---

## Takeaway

👉 *From satellite catalog → aligned data → AI prediction*

---

## Resources

- [TerraMind overview](https://ibm.github.io/terramind/)
- [TerraTorch TerraMind guid](https://terrastackai.github.io/terratorch/1.2.6/guide/terramind/)
- [STAC / Planetary Computer tutorial](https://stacspec.org/en/tutorials/reading-stac-planetary-computer/) 
- [Raster reprojection with Rasterio](https://rasterio.readthedocs.io/en/stable/topics/reproject.html) 
- [Sen1Floods11 benchmark description](https://docs.lxp.lu/web_services/open_ondemand/howtojlab/) 
- [MeluXina Open OnDemand JupyterLab docs](https://rasterio.readthedocs.io/en/stable/topics/reproject.html) 
- [MeluXina Open OnDemand JupyterLab](https://docs.lxp.lu/web_services/open_ondemand/jlab_env/) 
