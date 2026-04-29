# SCynergy 2026 - Operational GeoAI for Scalable Satellite Analytics

This repository contains the **materials and hands-on tutorials** for the SCynergy 2026 training workshop on **“Operational GeoAI for Scalable Satellite Analytics with Foundation Models”**, delivered by LuxProvide on the **MeluXina EuroHPC supercomputer**.

***

## Overview

The goal of this tutorial workshop is to introduce participants to **operational GeoAI workflows** by demonstrating how to **fine-tune and deploy geospatial foundation models (GeoFMs)** for large-scale satellite data analytics on GPU-accelerated HPC infrastructure.

The workshop combines:

*   Practical **GeoAI concepts**
*   **Hands-on exercises** on MeluXina GPUs
*   Best practices for **scalable workflows**

***

## Workshop Context

*   **Event:** SCynergy 2026
*   **Infrastructure:** MeluXina (EuroHPC JU supercomputer)
*   **Focus:** GPU-accelerated GeoAI and scalable satellite analytics
*   **Provider:** LuxProvide – Supercomputing Application Services group

This repository is primarily intended for **training and educational purposes** and is not designed as a standalone production library.

***

## Who Is This Workshop For?

This workshop is designed for:

*   Researchers and engineers working with **Earth Observation (EO)** or **satellite imagery**
*   Data scientists interested in **GeoAI and foundation models**
*   Users with basic experience in:
    *   Python
    *   Machine learning / deep learning

Prior experience with HPC systems or MeluXina is **helpful but not required**.

***

## Learning Objectives

By the end of this workshop, participants will be able to:

*   Understand the role of **geospatial foundation models** in EO analytics
*   Fine-tune a GeoFM for downstream geospatial tasks
*   Run GPU-accelerated training jobs on MeluXina (or other HPC or clouds)
*   Organize scalable GeoAI workflows suitable for large satellite datasets
*   Understand key challenges of **operationalizing GeoAI on HPC systems**

***

## Repository Structure

    SCynergy2026-GeoAI/
    ├── hands-on/
    │   ├── 00_installation.ipynb                       # Environment setup
    │   ├── 01_multimodal_data_acquisition_lux.ipynb    # Introductory / data configuration
    │   ├── 02_multimodal_data_packaging_lux.ipynb      # Data downloading and packaging
    │   ├── 03_multimodal_inference_lux.ipynb           # Multimodal inference and evaluation 
    │   ├── 04_terramind_v1_small_sen1floods11.ipynb    # (Optional) TerraMind model fine-tuning 
    ├── docs/                           # Training workshop documentation
    │   ├── index.md                    # Workshop introduction
    │   ├── connect_meluxina.md         # Instruction on how to connect to MeluXina (HPC)
    │   ├── worksheet.md                # Questions to answer for tutorials
    │   ├── cheat sheet.md              # Useful information for tutorials
    │   ├── faq.md                      # FAQ
    │   ├── 01_multimodal_data_acquisition_lux.md       # Notebook 01
    │   ├── 02_multimodal_data_packaging_lux.md         # Notebook 02
    │   ├── 03_multimodal_inference_lux.md              # Notebook 03
    │   └── 04_terramind_v1_small_sen1floods11.md       # Notebook 04 (Optional)
    ├── overrides/          # MkDocs theme overrides
    ├── .github/workflows/  # CI workflow for documentation deployment
    ├── mkdocs.yml          # MkDocs configuration
    ├── requirements.txt    # Python dependencies
    ├── LICENSE            
    └── README.md

***

## Getting Started

1. Read the workshop description from the webpage: https://luxprovide.github.io/SCynergy2026-GeoAI/ 
2. Clone or download notebook files (.ipynb) from the ["hands-on" directory](https://github.com/LuxProvide/SCynergy2026-GeoAI/tree/main/hands-on).
3. Connect to MeluXina via [Open OnDemand](https://portal.lxp.lu) and then run its JupyterLab app. Or, run the notebooks in other compatible environments.

***

## Acknowledgements

This workshop and its materials were developed by the  
**Supercomputing Application Services group at LuxProvide**.

© 2026 LuxProvide — All rights reserved.