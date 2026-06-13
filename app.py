import os
import io
import time
import glob
import re
import base64
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Classes definitions
CLASSES = ['Black Rot', 'ESCA', 'Healthy', 'Leaf Blight']
NUM_CLASSES = len(CLASSES)

# OOD & Confidence threshold configurations
OOD_THRESHOLD = 0.60
LOW_CONFIDENCE_THRESHOLD = 0.70

# Energy-based OOD Detection thresholds (Disabled to prevent false positives on real leaves)
# Max logit dari gambar OOD (CIFAR-100) berkisar 0.4-3.8
# Gambar daun anggur asli diharapkan menghasilkan max logit >> 5.0
MAX_LOGIT_THRESHOLD = 0.0
# Energy score: semakin tinggi (mendekati 0) = semakin OOD
# OOD range: -3.8 s/d -1.6 | ID diharapkan: < -5.0
ENERGY_THRESHOLD = 0.0

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
MAX_LOADING_LOGS = 100

def log_load(msg):
    print(msg)
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S")
    model_loading_logs.append(f"[{timestamp}] {msg}")
    if len(model_loading_logs) > MAX_LOADING_LOGS:
        del model_loading_logs[:len(model_loading_logs) - MAX_LOADING_LOGS]

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
                fold_model = timm.create_model('mobilevit_s', pretrained=False, num_classes=NUM_CLASSES)
                
                # Load state dict safely with PyTorch 2.6+ weights_only=True
                state_dict = torch.load(path, map_location=device, weights_only=True)
                
                # Check target classes dynamically from weight shape if mismatch
                if 'head.fc.weight' in state_dict:
                    weight_classes = state_dict['head.fc.weight'].shape[0]
                    if weight_classes != NUM_CLASSES:
                        log_load(f"Warning: model classes ({weight_classes}) tidak cocok dengan NUM_CLASSES ({NUM_CLASSES}). Menyesuaikan...")
                        fold_model = timm.create_model('mobilevit_s', pretrained=False, num_classes=weight_classes)
                
                fold_model.load_state_dict(state_dict)
                fold_model.to(device)
                fold_model.eval()
                
                models[fold_name] = fold_model
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
        
        with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
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

# Simulated prediction generator
def simulate_prediction(image):
    # Dynamic simulation based on average color of the image
    img = image.resize((32, 32)).convert('RGB')
    pixels = np.asarray(img, dtype=np.float32).reshape(-1, 3)
    avg_r = float(np.mean(pixels[:, 0]))
    avg_g = float(np.mean(pixels[:, 1]))
    avg_b = float(np.mean(pixels[:, 2]))
    
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



def validate_upload_file(file):
    if file is None or file.filename == '':
        raise ValueError('Nama file kosong')

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
        raise ValueError('Format file tidak didukung. Gunakan JPG, PNG, JPEG, BMP, atau WEBP.')

    image_bytes = file.read()
    if not image_bytes or len(image_bytes) == 0:
        raise ValueError('File gambar kosong.')

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()
    except Exception as exc:
        raise ValueError('File bukan gambar yang valid.') from exc

    return image_bytes


def is_plant_by_color(image, min_plant_pct=0.05):
    """
    Checks if the image has a minimum percentage of plant-like colors (green, yellow, brown).
    """
    # Resize image to speed up color analysis
    img_small = image.resize((100, 100))
    hsv_img = img_small.convert('HSV')
    hsv_np = np.array(hsv_img)
    
    h = hsv_np[:, :, 0]
    s = hsv_np[:, :, 1]
    v = hsv_np[:, :, 2]
    
    # PIL Hue: 0-255 (maps to 0-360 degrees)
    # Saturation & Value: 0-255
    # Green: 25 to 75 (approx 35 to 105 degrees)
    # Yellow/Brown: 8 to 25 (approx 11 to 35 degrees)
    # Require Saturation & Value >= 30 to avoid neutral background colors
    green_mask = (h >= 25) & (h <= 75) & (s >= 30) & (v >= 30)
    yellow_brown_mask = (h >= 8) & (h < 25) & (s >= 30) & (v >= 30)
    
    plant_mask = green_mask | yellow_brown_mask
    plant_pct = np.sum(plant_mask) / h.size
    
    return bool(plant_pct >= min_plant_pct), float(plant_pct)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "Tidak ada file gambar yang diunggah"}), 400

    file = request.files['file']

    try:
        image_bytes = validate_upload_file(file)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Color-based botanical check
        is_plant_leaf, plant_pct = is_plant_by_color(image)
        
        start_time = time.time()
        
        # Active folds used in prediction
        folds_used = list(models.keys())
        
        if model_loaded:
            # Prepare tensor
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            # Collect per-fold logits and probabilities for ensemble + OOD analysis
            all_fold_logits = []
            all_fold_probs = []
            with torch.no_grad():
                for fold_name, fold_model in models.items():
                    logits = fold_model(input_tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                    all_fold_logits.append(logits[0])  # shape (NUM_CLASSES,)
                    all_fold_probs.append(probs)
            
            # Average probabilities (Soft Voting Ensemble)
            avg_probs = torch.stack(all_fold_probs).mean(dim=0).cpu().numpy()
            
            # Average logits for energy-based OOD detection
            avg_logits = torch.stack(all_fold_logits).mean(dim=0)
            
            # Energy score: E(x) = -logsumexp(logits)
            # Lower (more negative) = more in-distribution
            # Higher (closer to 0) = more OOD
            energy_score = -torch.logsumexp(avg_logits, dim=0).item()
            
            # Max logit: magnitude of the strongest class activation
            # Low max logit = model tidak mengenali pola apapun = OOD
            max_logit_value = float(avg_logits.max().item())
            
            # Find class index with max probability
            pred_idx = int(avg_probs.argmax())
            confidence_score = float(avg_probs[pred_idx])
            
            # Map all probabilities
            all_probs = {}
            for i, p in enumerate(avg_probs):
                class_label = CLASSES[i] if i < len(CLASSES) else f"Class {i}"
                all_probs[class_label] = float(p)
            
            # ---- Multi-Signal OOD Detection ----
            # OOD jika max logit terlalu rendah (model tidak mengenali pola)
            # ATAU energy score terlalu tinggi (mendekati 0)
            # ATAU confidence score terlalu rendah (di bawah OOD_THRESHOLD)
            # ATAU deteksi warna menyatakan gambar bukan daun/tanaman
            ood_by_energy = energy_score > ENERGY_THRESHOLD
            ood_by_logit = max_logit_value < MAX_LOGIT_THRESHOLD
            ood_by_confidence = confidence_score < OOD_THRESHOLD
            ood_by_color = not is_plant_leaf
            
            # Gambar dianggap OOD jika salah satu sinyal deteksi menyatakan OOD
            is_ood = ood_by_logit or ood_by_energy or ood_by_confidence or ood_by_color
            is_low_confidence = (not is_ood) and confidence_score < LOW_CONFIDENCE_THRESHOLD
            
            # Generate Grad-CAM++ hanya jika BUKAN OOD (hemat waktu komputasi)
            if not is_ood:
                first_model = list(models.values())[0]
                gradcam_b64 = generate_gradcam_plusplus(first_model, input_tensor, image, pred_idx)
            else:
                gradcam_b64 = None
            
        else:
            # Simulation Mode
            time.sleep(0.4)  # Simulate network latency
            sim_probs = simulate_prediction(image)
            
            pred_idx = sim_probs.index(max(sim_probs))
            confidence_score = sim_probs[pred_idx]
            
            all_probs = {CLASSES[i]: float(sim_probs[i]) for i in range(len(CLASSES))}
            
            # Simulation mode intentionally avoids claiming real Grad-CAM output.
            gradcam_b64 = None
            
        # Determine OOD for simulation mode (fallback to confidence or color check)
        if not model_loaded:
            is_ood = (confidence_score < OOD_THRESHOLD) or (not is_plant_leaf)
            is_low_confidence = (not is_ood) and confidence_score < LOW_CONFIDENCE_THRESHOLD
        # is_ood and is_low_confidence already set above for model_loaded mode
        
        pred_class = 'Citra Tidak Dikenal' if is_ood else (CLASSES[pred_idx] if pred_idx < len(CLASSES) else f"Class {pred_idx}")
        
        inference_time = (time.time() - start_time) * 1000
        
        # Build response
        response_data = {
            "success": True,
            "class": pred_class,
            "confidence": f"{confidence_score * 100:.1f}%",
            "confidence_score": confidence_score,
            "all_probs": all_probs,
            "gradcam": gradcam_b64,
            "folds_used": folds_used,
            "inference_time": inference_time,
            "simulated": not model_loaded,
            "is_ood": is_ood,
            "is_low_confidence": is_low_confidence
        }
        
        # Tambahkan detail OOD metrics (untuk debugging dan reporting)
        if model_loaded:
            response_data["ood_details"] = {
                "energy_score": energy_score,
                "max_logit": max_logit_value,
                "energy_threshold": ENERGY_THRESHOLD,
                "max_logit_threshold": MAX_LOGIT_THRESHOLD,
                "ood_by_energy": ood_by_energy,
                "ood_by_logit": ood_by_logit,
                "ood_by_confidence": ood_by_confidence
            }
        
        return jsonify(response_data)
        
    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)