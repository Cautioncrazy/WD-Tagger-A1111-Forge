# Forge WD Tagger 🏷️

A high-performance image interrogation and tagging extension for **Stable Diffusion WebUI Forge**, inspired by the UI/UX of the [Civitai Browser](https://github.com/eduardoabreu81/sd-civitai-browser-ex) and powered by [SmilingWolf's WD Tagger](https://huggingface.co/spaces/SmilingWolf/wd-tagger).

[![Forge Compatibility](https://img.shields.io/badge/WebUI-Forge-orange.svg)](https://github.com/lllyasviel/stable-diffusion-webui-forge)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<img width="1920" height="1080" alt="Screenshot 2026-03-08 160142" src="https://github.com/user-attachments/assets/7d731117-9e3f-4d3b-8bde-b44f53e9a856" />

## ✨ Features

* **Civitai-Style Interface:** A modern, split-pane UI. Browse your local generations on the left; view and edit interrogated tags on the right.
* **Optimized for Forge:** Built to utilize the `onnxruntime` backend for near-instant tagging without interfering with Forge's VRAM management.
* **Latest Models:** Support for `WD SwinV2 V3`, `WD ViT V3`, and legacy `V2` models.
* **Batch Processing:** Interrogate entire folders of images and automatically save `.txt` sidecar files for LoRA training or archival.
* **Smart Filtering:** * Adjustable confidence thresholds for general and character tags.
    * Blacklist system to filter out unwanted tags (e.g., "rating: safe", "monochrome").
    * Optional escape characters for parentheses (Booru-style).

## 🚀 Installation

1. Open **SD WebUI Forge**.
2. Navigate to the **Extensions** tab -> **Install from URL**.
3. Paste the URL of this repository:
   `https://github.com/Cautioncrazy/WD-Tagger-A1111-Forge.git`
4. Click **Install**.
5. **Restart UI** (or close and relaunch the terminal).

> **Note:** On first run, the extension will download the required ONNX model files (~100-200MB) from Hugging Face.

## 🛠️ How to Use

### Single Image Mode
1. Go to the **WD Tagger** tab.
2. Select an image from your `outputs` gallery or drag-and-drop a file.
3. Adjust the **Threshold** (Default: `0.35`).
4. Click **Interrogate**.
5. Use the **Send to Prompt** buttons to instantly move tags to `txt2img` or `img2img`.

### Batch Mode
1. Switch to the **Batch Interrogator** sub-tab.
2. Input your source folder path.
3. Configure your tag suffix (e.g., `.txt`).
4. Click **Run Batch Interrogation**.

## 📊 Comparison

| Feature | Standard WD14 Tagger | Forge WD Tagger |
| :--- | :---: | :---: |
| **UI Layout** | Single Column | **Civitai-style Gallery** |
| **Backend** | TensorFlow/ONNX | **Optimized ONNX** |
| **Integration** | Standalone | **Deep Forge Hook** |
| **Mass Tagging** | Basic | **Advanced w/ Preview** |

## 🤝 Credits

* **UI Inspiration:** [eduardoabreu81's Civitai Browser](https://github.com/eduardoabreu81/sd-civitai-browser-ex)
* **Tagging Logic:** [SmilingWolf](https://huggingface.co/SmilingWolf) for the incredible WD Tagger models.
* **Platform:** [lllyasviel](https://github.com/lllyasviel) for SD WebUI Forge.

---
