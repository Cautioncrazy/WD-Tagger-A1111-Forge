import launch

if not launch.is_installed("huggingface_hub"):
    launch.run_pip("install huggingface-hub", "requirements for Forge-WD-Tagger")

if not launch.is_installed("cv2"):
    launch.run_pip("install opencv-python", "requirements for Forge-WD-Tagger")

if not launch.is_installed("pandas"):
    launch.run_pip("install pandas", "requirements for Forge-WD-Tagger")

if not launch.is_installed("onnxruntime") and not launch.is_installed("onnxruntime-gpu"):
    import torch
    if torch.cuda.is_available():
        launch.run_pip("install onnxruntime-gpu", "requirements for Forge-WD-Tagger (onnxruntime-gpu)")
    else:
        launch.run_pip("install onnxruntime", "requirements for Forge-WD-Tagger (onnxruntime)")

if not launch.is_installed("clip_interrogator"):
    launch.run_pip("install clip-interrogator", "requirements for Forge-WD-Tagger (CLIP-Interrogator)")
