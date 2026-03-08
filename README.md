# Forge WD Tagger 🏷️

A high-performance image interrogation and tagging extension for **Stable Diffusion WebUI Forge**, inspired by the UI/UX of the [Civitai Browser](https://github.com/eduardoabreu81/sd-civitai-browser-ex). It is powered by [SmilingWolf's WD Tagger](https://huggingface.co/spaces/SmilingWolf/wd-tagger) for Booru-style tags and the [CLIP Interrogator 2 API](https://huggingface.co/spaces/fffiloni/CLIP-Interrogator-2) for rich, natural language prompting.

[![Forge Compatibility](https://img.shields.io/badge/WebUI-Forge-orange.svg)](https://github.com/lllyasviel/stable-diffusion-webui-forge)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**WD Tagger** 
<img width="1816" height="1016" alt="Screenshot 2026-03-08 160142" src="https://github.com/user-attachments/assets/d8b039cb-fbf1-4ecd-b933-6bb92c3daabe" /> 
**Batch Interrogator**
<img width="1920" height="1080" alt="Screenshot 2026-03-08 162823" src="https://github.com/user-attachments/assets/d552fce0-503c-4854-87a9-a9c90ab88e05" />
**CLIP Interrogator**
<img width="1812" height="911" alt="Screenshot 2026-03-08 162107" src="https://github.com/user-attachments/assets/ed48d290-956f-4845-8a6a-97ab0931e6ac" />


## ✨ Features

* **Civitai-Style Interface:** A modern, split-pane UI. Browse your local generations on the left; view and edit interrogated tags on the right.
* **Dual Interrogation Modes:** Seamlessly switch between WD14 (for precise anime/booru tags) and CLIP Interrogator 2 (for highly detailed, natural language style and composition prompts).
* **Optimized for Forge:** Built to utilize the `onnxruntime` backend for near-instant WD tagging without interfering with Forge's VRAM management.
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

> **Note:** On first run, the WD Tagger will download its required ONNX model files (~100-200MB) from Hugging Face. The CLIP Interrogator utilizes a lightweight API call to HuggingFace Spaces.

## 🛠️ How to Use

### WD Tagger Mode (Booru Tags)
1. Go to the **WD Tagger** tab.
2. Select an image from your `outputs` gallery or drag-and-drop a file.
3. Adjust the **Threshold** (Default: `0.35`).
4. Click **Interrogate**.
5. Use the **Send to Prompt** buttons to instantly move tags to `txt2img` or `img2img`.

### CLIP Interrogator Mode (Natural Language)
1. Switch to the **CLIP Interrogator** sub-tab.
2. Upload your image.
3. Click **Interrogate**. The extension will securely ping the HuggingFace Space API to analyze your image's style, medium, and metadata.
4. Receive a perfectly structured, descriptive prompt ideal for realistic or heavily stylized SDXL/SD 1.5 checkpoints.

### Batch Mode
1. Switch to the **Batch Interrogator** sub-tab.
2. Input your source folder path.
3. Configure your tag suffix (e.g., `.txt`).
4. Click **Run Batch Interrogation**.

## 📊 Comparison

| Feature | Standard WD14 Tagger | Forge WD Tagger |
| :--- | :---: | :---: |
| **UI Layout** | Single Column | **Civitai-style Gallery** |
| **Interrogation** | Booru Tags Only | **WD14 + CLIP Interrogator 2** |
| **Backend** | TensorFlow/ONNX | **Optimized ONNX + API** |
| **Integration** | Standalone | **Deep Forge Hook** |

## 🤝 Credits

* **UI Inspiration:** [eduardoabreu81's Civitai Browser](https://github.com/eduardoabreu81/sd-civitai-browser-ex)
* **Booru Tagging Logic:** [SmilingWolf](https://huggingface.co/SmilingWolf) for the incredible WD Tagger models.
* **CLIP Interrogation:** [fffiloni](https://huggingface.co/spaces/fffiloni/CLIP-Interrogator-2) for hosting the highly accurate CLIP Interrogator 2 Space.
* **Platform:** [lllyasviel](https://github.com/lllyasviel) for SD WebUI Forge.

---
