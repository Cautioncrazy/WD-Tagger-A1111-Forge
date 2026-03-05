import os
import csv
import json
import logging
import cv2
import numpy as np
import pandas as pd
import gradio as gr
from PIL import Image
from huggingface_hub import hf_hub_download
import onnxruntime as ort

import modules.scripts as scripts
from modules import script_callbacks, shared
import modules.generation_parameters_copypaste as parameters_copypaste

logger = logging.getLogger("forge-wd-browser")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

MODELS_DIR = os.path.join(scripts.basedir(), "models", "WD-Tagger")
DEFAULT_GALLERY_DIR = os.path.join(shared.data_path, "outputs", "txt2img-images")

# Define available models
AVAILABLE_MODELS = {
    "wd-swinv2-tagger-v3": "SmilingWolf/wd-swinv2-tagger-v3",
    "wd-vit-tagger-v3": "SmilingWolf/wd-vit-tagger-v3",
    "wd-v1-4-moat-tagger-v2": "SmilingWolf/wd-v1-4-moat-tagger-v2",
}

# --- Backend Logic ---

class WDTagger:
    def __init__(self):
        self.model_name = None
        self.model = None
        self.tags = []
        self.character_tags = []
        self.general_tags = []
        self.rating_tags = []
        self.target_size = 448 # Default for V3

    def load_model(self, model_name):
        if self.model_name == model_name and self.model is not None:
            return True # Already loaded

        repo_id = AVAILABLE_MODELS.get(model_name)
        if not repo_id:
            logger.error(f"Model {model_name} not found in available models.")
            return False

        logger.info(f"Loading WD Tagger model: {model_name} from {repo_id}")
        os.makedirs(MODELS_DIR, exist_ok=True)

        try:
            # Download model and tags
            model_path = hf_hub_download(repo_id=repo_id, filename="model.onnx", cache_dir=MODELS_DIR)
            tags_path = hf_hub_download(repo_id=repo_id, filename="selected_tags.csv", cache_dir=MODELS_DIR)

            # Load ONNX session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
            self.model = ort.InferenceSession(model_path, providers=providers)

            # Load tags
            df = pd.read_csv(tags_path)
            self.tags = df['name'].tolist()

            self.general_tags = df[df['category'] == 0]['name'].tolist()
            self.character_tags = df[df['category'] == 4]['name'].tolist()
            self.rating_tags = df[df['category'] == 9]['name'].tolist()

            # Get expected input size
            input_info = self.model.get_inputs()[0]
            if len(input_info.shape) == 4:
                self.target_size = input_info.shape[2] # NCHW or NHWC

            self.model_name = model_name
            return True

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False

    def preprocess_image(self, image: Image.Image):
        image = image.convert("RGB")
        image = np.array(image)
        # BGR
        image = image[:, :, ::-1]

        # Padding to square
        old_size = image.shape[:2]
        ratio = float(self.target_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        image = cv2.resize(image, (new_size[1], new_size[0]))

        delta_w = self.target_size - new_size[1]
        delta_h = self.target_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        image = image.astype(np.float32)

        # HWC to CHW if needed by model. Usually WD uses NHWC for onnx but let's check
        input_info = self.model.get_inputs()[0]
        if input_info.shape[1] == 3: # NCHW
             image = np.expand_dims(image.transpose(2, 0, 1), 0)
        else: # NHWC
             image = np.expand_dims(image, 0)

        return image

    def interrogate(self, image: Image.Image, gen_thresh=0.35, char_thresh=0.35, exclude_tags="", escape_parens=False):
        if self.model is None:
            return {"error": "Model not loaded"}

        image_tensor = self.preprocess_image(image)

        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name

        preds = self.model.run([output_name], {input_name: image_tensor})[0][0]

        tag_scores = dict(zip(self.tags, preds))

        excluded = [t.strip() for t in exclude_tags.split(",") if t.strip()]

        def filter_tags(tag_list, thresh):
            res = {}
            for t in tag_list:
                if t in excluded: continue
                score = tag_scores.get(t, 0)
                if score >= thresh:
                    formatted_tag = t.replace('_', ' ')
                    if escape_parens:
                        formatted_tag = formatted_tag.replace('(', r'\(').replace(')', r'\)')
                    res[formatted_tag] = float(score)
            return dict(sorted(res.items(), key=lambda item: item[1], reverse=True))

        general_res = filter_tags(self.general_tags, gen_thresh)
        char_res = filter_tags(self.character_tags, char_thresh)

        # Rating uses argmax
        rating_res = {t: float(tag_scores.get(t, 0)) for t in self.rating_tags}
        best_rating = max(rating_res, key=rating_res.get)

        return {
            "general": general_res,
            "character": char_res,
            "rating": {best_rating: rating_res[best_rating]}
        }

tagger = WDTagger()

# --- UI Functions ---

def load_gallery(path):
    if not os.path.exists(path):
        return [], f"Directory not found: {path}"

    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    images = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(valid_extensions):
                images.append(os.path.join(root, file))

    return images, f"Found {len(images)} images in {path}"

def on_interrogate(input_mode, uploaded_image, selected_gallery_image, model_name, gen_thresh, char_thresh, exclude_tags, escape_parens):
    image = uploaded_image if input_mode == "Upload Image" else selected_gallery_image

    if image is None:
         return "No image selected.", "", "", ""

    if not tagger.load_model(model_name):
         return "Failed to load model.", "", "", ""

    pil_image = Image.fromarray(image)
    res = tagger.interrogate(pil_image, gen_thresh, char_thresh, exclude_tags, escape_parens)

    if "error" in res:
        return res["error"], "", "", ""

    gen_str = ", ".join(res["general"].keys())
    char_str = ", ".join(res["character"].keys())
    rating_str = ", ".join([f"{k} ({v:.2f})" for k, v in res["rating"].items()])

    all_tags = []
    if char_str: all_tags.append(char_str)
    if gen_str: all_tags.append(gen_str)

    full_prompt = ", ".join(all_tags)

    return full_prompt, gen_str, char_str, rating_str

def on_batch_process(folder_path, model_name, gen_thresh, char_thresh, exclude_tags, escape_parens, conflict_action, tag_suffix):
    if not os.path.exists(folder_path):
        return f"Directory not found: {folder_path}"

    if not tagger.load_model(model_name):
         return "Failed to load model."

    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    images = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(valid_extensions):
                images.append(os.path.join(root, file))

    if not images:
        return f"No images found in {folder_path}"

    processed = 0
    skipped = 0

    for img_path in images:
        txt_path = os.path.splitext(img_path)[0] + tag_suffix

        if os.path.exists(txt_path):
            if conflict_action == "Skip":
                skipped += 1
                continue

        try:
            with Image.open(img_path) as pil_image:
                res = tagger.interrogate(pil_image, gen_thresh, char_thresh, exclude_tags, escape_parens)

            gen_str = ", ".join(res["general"].keys())
            char_str = ", ".join(res["character"].keys())

            all_tags = []
            if char_str: all_tags.append(char_str)
            if gen_str: all_tags.append(gen_str)
            full_prompt = ", ".join(all_tags)

            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    existing_tags = f.read().strip()

                if conflict_action == "Overwrite":
                    final_tags = full_prompt
                elif conflict_action == "Prepend":
                    final_tags = f"{full_prompt}, {existing_tags}" if existing_tags else full_prompt
                elif conflict_action == "Append":
                     final_tags = f"{existing_tags}, {full_prompt}" if existing_tags else full_prompt
            else:
                final_tags = full_prompt

            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(final_tags)

            processed += 1

        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")

    return f"Batch processing complete. Processed {processed} images. Skipped {skipped} images."

# --- UI Setup ---

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as wd_browser_interface:
        with gr.Tabs():
            # --- Single Image Mode Tab ---
            with gr.TabItem("Single Image Interrogation"):
                with gr.Row():
                    # Left Side: Gallery & Selection
                    with gr.Column(scale=1):
                        gr.Markdown("### Image Source")
                        input_mode = gr.Radio(choices=["Upload Image", "Folder Gallery"], value="Upload Image", label="Mode")

                        with gr.Group(visible=True) as upload_group:
                            uploaded_image = gr.Image(label="Upload Image", type="numpy", interactive=True)

                        with gr.Group(visible=False) as gallery_group:
                            gallery_path = gr.Textbox(label="Gallery Path", value=DEFAULT_GALLERY_DIR, interactive=True)
                            refresh_btn = gr.Button("Refresh Gallery")
                            status_msg = gr.Markdown()

                            gallery = gr.Gallery(label="Images", show_label=False, columns=3, height=500, object_fit="contain", interactive=False)

                        selected_gallery_image = gr.Image(label="Selected Image", type="numpy", visible=False)

                        def toggle_mode(mode):
                            return gr.update(visible=mode == "Upload Image"), gr.update(visible=mode == "Folder Gallery")

                        input_mode.change(
                            fn=toggle_mode,
                            inputs=[input_mode],
                            outputs=[upload_group, gallery_group]
                        )

                        def select_image(evt: gr.SelectData, gallery_files):
                            # The gallery currently returns lists of tuples/dicts depending on gradio version.
                            # Usually evt.index is reliable
                            if gallery_files and evt.index < len(gallery_files):
                                # gallery_files might be a list of dicts {'name': path} or tuples (path, label)
                                item = gallery_files[evt.index]
                                if isinstance(item, dict):
                                    path = item.get('name')
                                elif isinstance(item, tuple) or isinstance(item, list):
                                    path = item[0]
                                else:
                                    path = item
                                return path
                            return None

                        gallery.select(fn=select_image, inputs=[gallery], outputs=[selected_gallery_image])

                        refresh_btn.click(
                            fn=load_gallery,
                            inputs=[gallery_path],
                            outputs=[gallery, status_msg]
                        )

                        # Load default gallery on startup
                        wd_browser_interface.load(
                            fn=load_gallery,
                            inputs=[gallery_path],
                            outputs=[gallery, status_msg]
                        )

                    # Right Side: Results & Controls
                    with gr.Column(scale=1):
                        gr.Markdown("### Interrogation Settings")
                        with gr.Row():
                            model_dropdown = gr.Dropdown(label="Model", choices=list(AVAILABLE_MODELS.keys()), value="wd-swinv2-tagger-v3")

                        with gr.Row():
                            gen_threshold = gr.Slider(label="General Threshold", minimum=0.0, maximum=1.0, value=0.35, step=0.01)
                            char_threshold = gr.Slider(label="Character Threshold", minimum=0.0, maximum=1.0, value=0.35, step=0.01)

                        exclude_tags = gr.Textbox(label="Exclude Tags (comma-separated)", placeholder="e.g. monochrome, rating: safe")
                        escape_parens = gr.Checkbox(label=r"Escape Parentheses `\(\)`", value=False)

                        interrogate_btn = gr.Button("Interrogate Image", variant="primary")

                        gr.Markdown("### Results")
                        full_prompt_output = gr.TextArea(label="Full Prompt", lines=4, interactive=True)

                        with gr.Row():
                            send_to_txt2img = gr.Button("Send to txt2img")
                            send_to_img2img = gr.Button("Send to img2img")

                        # JS injection for sending to prompt
                        send_to_txt2img.click(
                            fn=None,
                            inputs=[full_prompt_output],
                            outputs=[],
                            js="(tags) => { const el = document.querySelector('#txt2img_prompt textarea'); if(el) { el.value = tags; el.dispatchEvent(new Event('input', {bubbles: true})); } }"
                        )
                        send_to_img2img.click(
                            fn=None,
                            inputs=[full_prompt_output],
                            outputs=[],
                            js="(tags) => { const el = document.querySelector('#img2img_prompt textarea'); if(el) { el.value = tags; el.dispatchEvent(new Event('input', {bubbles: true})); } }"
                        )

                        with gr.Accordion("Detailed Tags", open=False):
                            rating_output = gr.Textbox(label="Rating", interactive=False)
                            char_output = gr.TextArea(label="Characters", interactive=False)
                            gen_output = gr.TextArea(label="General Tags", interactive=False)

                        interrogate_btn.click(
                            fn=on_interrogate,
                            inputs=[input_mode, uploaded_image, selected_gallery_image, model_dropdown, gen_threshold, char_threshold, exclude_tags, escape_parens],
                            outputs=[full_prompt_output, gen_output, char_output, rating_output]
                        )

            # --- Batch Interrogator Tab ---
            with gr.TabItem("Batch Interrogator"):
                gr.Markdown("### Mass Tagging Mode")
                gr.Markdown("Batch process a folder and save `.txt` files alongside images.")

                batch_folder_path = gr.Textbox(label="Folder Path", placeholder="Path to folder containing images")
                batch_model_dropdown = gr.Dropdown(label="Model", choices=list(AVAILABLE_MODELS.keys()), value="wd-swinv2-tagger-v3")

                with gr.Row():
                    batch_gen_threshold = gr.Slider(label="General Threshold", minimum=0.0, maximum=1.0, value=0.35, step=0.01)
                    batch_char_threshold = gr.Slider(label="Character Threshold", minimum=0.0, maximum=1.0, value=0.35, step=0.01)

                batch_exclude_tags = gr.Textbox(label="Exclude Tags (comma-separated)")
                batch_escape_parens = gr.Checkbox(label=r"Escape Parentheses `\(\)`", value=False)

                with gr.Row():
                    conflict_action = gr.Radio(label="If .txt file already exists:", choices=["Skip", "Overwrite", "Prepend", "Append"], value="Skip")
                    tag_suffix = gr.Textbox(label="Tag file suffix", value=".txt")

                run_batch_btn = gr.Button("Run Batch Interrogation", variant="primary")
                batch_status_output = gr.Textbox(label="Status", interactive=False)

                run_batch_btn.click(
                    fn=on_batch_process,
                    inputs=[batch_folder_path, batch_model_dropdown, batch_gen_threshold, batch_char_threshold, batch_exclude_tags, batch_escape_parens, conflict_action, tag_suffix],
                    outputs=[batch_status_output]
                )

    return [(wd_browser_interface, "WD Browser", "forge_wd_browser")]

script_callbacks.on_ui_tabs(on_ui_tabs)
