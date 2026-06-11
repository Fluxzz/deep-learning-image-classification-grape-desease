import sys
import os
import requests
import json

def test_predict(image_path, url="http://127.0.0.1:7860/predict"):
    """
    Sends an image file to the Flask /predict endpoint and prints the result.
    """
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' tidak ditemukan.")
        sys.exit(1)
        
    print(f"Mengirim {image_path} ke {url}...")
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            response = requests.post(url, files=files)
        
        print(f"Status Code: {response.status_code}")
        print("Response JSON:")
        try:
            print(json.dumps(response.json(), indent=2))
        except json.JSONDecodeError:
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print(f"Error: Gagal terhubung ke {url}. Pastikan server Flask berjalan.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Penggunaan: python test_predict.py <path_ke_gambar> [url_api]")
        print("Contoh: python test_predict.py sample_leaf.jpg")
        sys.exit(1)
        
    img_path = sys.argv[1]
    api_url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:7860/predict"
    test_predict(img_path, api_url)
