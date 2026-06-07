import app
import torch
from PIL import Image, ImageDraw
import io

# Create a dummy image resembling a blood cell (red circle on light background)
img = Image.new('RGB', (300, 300), color=(240, 240, 250))
draw = ImageDraw.Draw(img)
draw.ellipse([80, 80, 220, 220], fill=(220, 80, 80), outline=(150, 40, 40), width=4)

# Prepare test request
with app.app.test_client() as client:
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    print("Sending mock request to /predict...")
    response = client.post('/predict', data={
        'file': (img_bytes, 'test.png')
    })
    
    print("Response Status Code:", response.status_code)
    data = response.get_json()
    if data:
        print("Response Success:", data.get('success'))
        print("Predicted Class:", data.get('class'))
        print("Confidence:", data.get('confidence'))
        print("Folds Used:", data.get('folds_used'))
        print("All Probs:", data.get('all_probs'))
        print("Grad-CAM base64 starts with:", data.get('gradcam')[:50] if data.get('gradcam') else "None")
        print("Inference Time (ms):", data.get('inference_time'))
    else:
        print("Error: No JSON data in response.")
