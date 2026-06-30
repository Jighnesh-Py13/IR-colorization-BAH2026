# Infrared Image Colorization & Super-Resolution
### Bhartiya Antriksh Hackathon (BAH) 2026 Submission

## Overview

This project presents a deep learning pipeline for enhancing Thermal Infrared (TIR) satellite imagery captured by Landsat 9.

The pipeline performs two sequential tasks:

1. **Super-Resolution**
   - Converts low-resolution 200m Thermal Infrared imagery into 100m resolution imagery using an ESPCN model.

2. **Colorization**
   - Converts the generated 100m Thermal Infrared image into a synthetic RGB representation using a U-Net architecture.

The objective is to improve the interpretability of thermal imagery while preserving spatial information.

---

## Pipeline

```
200m TIR (B10)
        │
        ▼
ESPCN Super-Resolution
        │
        ▼
100m TIR
        │
        ▼
U-Net Colorization
        │
        ▼
100m RGB (Blue-Green-Red)
```

---

## Project Structure

```
IR-colorization-BAH2026/
│
├── input/
├── output/
│   ├── patches/
│   └── model_outputs/
│
├── src/
│   ├── datasets.py
│   ├── inference.py
│   ├── train_sr.py
│   ├── train_color.py
│   └── models/
│       ├── super_resolution.py
│       └── colorization.py
│
├── weights/
│   ├── espcn.pth
│   └── color_unet.pth
│
├── driver.py
└── README.md
```

---

## Models Used

### Super-Resolution

- ESPCN (Efficient Sub-Pixel Convolutional Neural Network)
- Upscaling factor: **2×**
- Input:
  - 1 × 256 × 256
- Output:
  - 1 × 512 × 512

---

### Colorization

- U-Net Encoder-Decoder
- Input:
  - 1 × 512 × 512 Thermal Infrared
- Output:
  - 3 × 512 × 512 RGB

---

## Training

### Super Resolution

```bash
python -m src.train_sr
```

### Colorization

```bash
python -m src.train_color
```

Both models were trained using PyTorch with normalized input data and L1 loss.

---

## Inference

```bash
python -m src.inference \
    --product_id <product_id> \
    --input_dir input/<product_id> \
    --sr_weights weights/espcn.pth \
    --color_weights weights/color_unet.pth
```

Generated outputs are stored in:

```
output/model_outputs/
```

```
tir_superresolved_100m/
```

```
colorized_tir_100m/
```

---

## Sample Results

The pipeline generates:

```
Raw 200m Thermal Image
        ↓
Super-Resolved 100m Thermal Image
        ↓
Synthetic RGB Image
```

*(See the technical report for qualitative results.)*

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Rasterio
- TIFFFile
- OpenCV
- Matplotlib

---

## Repository Contents

- Source code
- Trained ESPCN model
- Trained U-Net model
- Inference pipeline
- Technical report
- Sample outputs

---

## Team T.A.R.S

**Bhartiya Antriksh Hackathon 2026 Submission**

Developed as part of the Infrared Image Colorization and Enhancement challenge.