import os
import io
import time
import glob
import re
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Classes definitions
CLASSES = ['Black Rot', 'ESCA', 'Healthy', 'Leaf Blight']
NUM_CLASSES = len(CLASSES)

# Image transform pipeline (MobileViT input: 224x224)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dynamic Ensemble Loading
models = {}
model_loaded = False
model_error = None
model_loading_logs = []

def log_load(msg):
    print(msg)
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S")
    model_loading_logs.append(f"[{timestamp}] {msg}")

log_load(f"Memulai inisialisasi ensemble model pada device: {device}")
try:
    import timm
    # Find all fold weights in the current directory matching 'best_mobilevit_fold*.pth'
    checkpoint_paths = glob.glob('best_mobilevit_fold*.pth')
    
    # Sort them by fold number
    checkpoint_paths.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
    
    if len(checkpoint_paths) > 0:
        log_load(f"Menemukan {len(checkpoint_paths)} checkpoint model folds: {checkpoint_paths}")
        for path in checkpoint_paths:
            fold_name = os.path.basename(path).replace('best_mobilevit_fold', 'Fold ').replace('.pth', '')
            try:
                # Initialize timm mobilevit_s model
                model = timm.create_model('mobilevit_s', pretrained=False, num_classes=NUM_CLASSES)
                
                # Load state dict
                state_dict = torch.load(path, map_location=device)
                
                # Check target classes dynamically from weight shape if mismatch
                if 'head.fc.weight' in state_dict:
                    weight_classes = state_dict['head.fc.weight'].shape[0]
                    if weight_classes != NUM_CLASSES:
                        log_load(f"Warning: model classes ({weight_classes}) tidak cocok dengan NUM_CLASSES ({NUM_CLASSES}). Menyesuaikan...")
                        model = timm.create_model('mobilevit_s', pretrained=False, num_classes=weight_classes)
                
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()
                
                models[fold_name] = model
                log_load(f"Sukses memuat model fold: {fold_name} dari {path}")
            except Exception as fold_err:
                log_load(f"Error memuat model fold {path}: {fold_err}")
        
        if len(models) > 0:
            model_loaded = True
            log_load(f"Ensemble model siap digunakan dengan {len(models)} folds.")
        else:
            model_error = "Gagal memuat seluruh checkpoint folds yang ditemukan."
            log_load(f"ERROR: {model_error}")
    else:
        model_error = "Tidak ada file checkpoint 'best_mobilevit_fold*.pth' yang ditemukan di direktori."
        log_load(f"ERROR: {model_error}")

except Exception as e:
    model_error = f"Error saat menyiapkan environment loading model: {e}"
    log_load(f"ERROR: {model_error}")


# Grad-CAM++ implementation
def generate_gradcam_plusplus(model, input_tensor, original_image, pred_idx):
    try:
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        
        # Target layer of mobilevit_s (last stage for coarse/attention maps)
        # Stages are the feature extractor blocks
        target_layers = [model.stages[-1]]
        
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
        
        # Run CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_idx)])[0]
        
        # Prepare original image as numpy [0, 1] RGB
        img_resized = original_image.resize((224, 224))
        img_np = np.array(img_resized, dtype=np.float32) / 255.0
        
        # Overlay heatmap on image
        cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True, image_weight=0.55)
        
        # Convert to PIL and save as base64
        cam_pil = Image.fromarray(cam_image)
        buffered = io.BytesIO()
        cam_pil.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"
        
    except Exception as e:
        print(f"Error generating Grad-CAM++: {e}")
        return None

# Mock Grad-CAM++ overlay generator for Simulation Mode
def generate_mock_gradcam(original_image, class_idx):
    try:
        img_resized = original_image.resize((224, 224)).convert('RGB')
        
        # Create a radial gradient for mock heatmap
        mask = Image.new('L', (224, 224), 0)
        draw = ImageDraw.Draw(mask)
        
        # Put focus coordinates depending on the class index to simulate variety
        focus_centers = [
            (112, 112),  # center
            (80, 100),   # upper-left
            (140, 120),  # lower-right
            (100, 130)   # lower-left
        ]
        cx, cy = focus_centers[class_idx % len(focus_centers)]
        
        draw.ellipse([cx - 50, cy - 50, cx + 50, cy + 50], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(25))
        
        # Apply colormap mapping manually (0->blue, 128->green, 255->red)
        heatmap = Image.new('RGB', (224, 224))
        heatmap_pixels = heatmap.load()
        mask_pixels = mask.load()
        for y in range(224):
            for x in range(224):
                v = mask_pixels[x, y]
                if v < 128:
                    r = 0
                    g = int(v * 2)
                    b = 255 - int(v * 2)
                else:
                    r = int((v - 128) * 2)
                    g = 255 - int((v - 128) * 2)
                    b = 0
                heatmap_pixels[x, y] = (r, g, b)
                
        # Blend heatmap with original image
        blended = Image.blend(img_resized, heatmap, alpha=0.45)
        buffered = io.BytesIO()
        blended.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        print(f"Error generating mock Grad-CAM: {e}")
        return None

# Simulated prediction generator
def simulate_prediction(image):
    # Dynamic simulation based on average color of the image
    img = image.resize((32, 32))
    pixels = list(img.getdata())
    avg_r = sum(p[0] for p in pixels) / len(pixels)
    avg_g = sum(p[1] for p in pixels) / len(pixels)
    avg_b = sum(p[2] for p in pixels) / len(pixels)
    
    # Generate mock probabilities for each class (Black Rot, ESCA, Healthy, Leaf Blight)
    if avg_r > avg_g and avg_r > avg_b:
        # Reddish: Black Rot / ESCA
        probs = [0.45, 0.35, 0.10, 0.10]
    elif avg_b > avg_r and avg_b > avg_g:
        # Blueish / Dark: Healthy
        probs = [0.10, 0.10, 0.65, 0.15]
    else:
        # Greenish / Neutral: Leaf Blight
        probs = [0.15, 0.15, 0.15, 0.55]
        
    return probs


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/status', methods=['GET'])
def status():
    expected_folds = 5
    loaded_folds = list(models.keys())
    return jsonify({
        "success": True,
        "model_loaded": model_loaded,
        "model_error": model_error,
        "device": str(device),
        "loaded_folds": loaded_folds,
        "total_folds_expected": expected_folds,
        "total_folds_loaded": len(loaded_folds),
        "loading_logs": model_loading_logs
    })



@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "Tidak ada file gambar yang diunggah"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "Nama file kosong"}), 400
        
    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        start_time = time.time()
        
        # Active folds used in prediction
        folds_used = list(models.keys())
        
        if model_loaded:
            # Prepare tensor
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            # Average probabilities across all fold models (Soft Voting Ensemble)
            outputs_sum = None
            with torch.no_grad():
                for fold_name, model in models.items():
                    logits = model(input_tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                    if outputs_sum is None:
                        outputs_sum = probs
                    else:
                        outputs_sum += probs
            
            avg_probs = (outputs_sum / len(models)).cpu().numpy()
            
            # Find class index with max probability
            pred_idx = int(avg_probs.argmax())
            confidence_score = float(avg_probs[pred_idx])
            pred_class = CLASSES[pred_idx] if pred_idx < len(CLASSES) else f"Class {pred_idx}"
            
            # Map all probabilities
            all_probs = {}
            for i, p in enumerate(avg_probs):
                class_label = CLASSES[i] if i < len(CLASSES) else f"Class {i}"
                all_probs[class_label] = float(p)
                
            # Generate Grad-CAM++ using the first fold model
            first_model = list(models.values())[0]
            gradcam_b64 = generate_gradcam_plusplus(first_model, input_tensor, image, pred_idx)
            
        else:
            # Simulation Mode
            time.sleep(0.4)  # Simulate network latency
            sim_probs = simulate_prediction(image)
            
            pred_idx = sim_probs.index(max(sim_probs))
            confidence_score = sim_probs[pred_idx]
            pred_class = CLASSES[pred_idx]
            
            all_probs = {CLASSES[i]: float(sim_probs[i]) for i in range(len(CLASSES))}
            
            # Generate mock Grad-CAM overlay
            gradcam_b64 = generate_mock_gradcam(image, pred_idx)
            
        inference_time = (time.time() - start_time) * 1000
        
        return jsonify({
            "success": True,
            "class": pred_class,
            "confidence": f"{confidence_score * 100:.1f}%",
            "confidence_score": confidence_score,
            "all_probs": all_probs,
            "gradcam": gradcam_b64,
            "folds_used": folds_used,
            "inference_time": inference_time,
            "simulated": not model_loaded
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
