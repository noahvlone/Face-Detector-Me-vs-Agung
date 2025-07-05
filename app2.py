import os
import warnings
from flask import Flask, render_template, request, redirect, url_for, jsonify
import cv2
import numpy as np
from skimage.feature import hog
import joblib
from datetime import datetime
import urllib.parse
import base64

app = Flask(__name__)

# Konfigurasi
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'jfif'}
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Batas 5MB

# Fungsi untuk memeriksa ekstensi file
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Memuat model dengan menekan peringatan
def load_model_safe(model_path):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return joblib.load(model_path)

# Memuat model
MODELS_DIR = 'models'
try:
    svm_model = load_model_safe(os.path.join(MODELS_DIR, 'svm_model.pkl'))
    knn_model = load_model_safe(os.path.join(MODELS_DIR, 'knn_model.pkl'))
    le = load_model_safe(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
except Exception as e:
    print(f"Error loading models: {e}")
    # Keluar jika model tidak bisa dimuat
    raise e

# Fungsi preprocessing dan ekstraksi fitur
def preprocess_image(img_path=None, img_array=None, target_size=(128, 128)):
    if img_path:
        img = cv2.imread(img_path)
    elif img_array is not None:
        img = img_array
    else:
        return None
        
    if img is None:
        return None
    
    # Preprocessing
    resized = cv2.resize(img, target_size)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    normalized = gray / 255.0
    
    # Ekstrak fitur HOG
    fd, _ = hog(
        normalized,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=True
    )
    return fd.reshape(1, -1)

# Route untuk halaman utama
@app.route('/')
def home():
    return render_template('index.html')

# Route untuk prediksi dari file upload
@app.route('/predict', methods=['POST'])
def predict():
    # Cek apakah ada file yang diupload
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    # Jika user tidak memilih file
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Buat nama file unik dengan timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            print(f"File saved to: {filepath}")
        except Exception as e:
            print(f"Error saving file: {e}")
            return "Gagal menyimpan file", 500
        
        # Preprocessing dan ekstraksi fitur
        features = preprocess_image(img_path=filepath)
        if features is None:
            return "Gagal memproses gambar. Pastikan file yang diunggah adalah gambar.", 400
        
        # Pilih model
        model_type = request.form.get('model', 'svm')
        if model_type == 'svm':
            prediction = svm_model.predict(features)
        else:  # knn
            prediction = knn_model.predict(features)
            
        # Decode label
        class_name = le.inverse_transform(prediction)[0]
        
        # Buat URL untuk gambar
        image_url = f"/static/uploads/{urllib.parse.quote(filename)}"
        print(f"Image URL: {image_url}")
        
        return render_template('result.html', 
                              image_url=image_url,
                              prediction=class_name,
                              model_used=model_type.upper())
    
    return "Format file tidak diizinkan. Gunakan format: png, jpg, jpeg, jfif", 400

# Route untuk prediksi dari gambar webcam
@app.route('/predict_webcam', methods=['POST'])
def predict_webcam():
    # Ambil data gambar dari request
    data = request.get_json()
    image_data = data['image']
    
    # Konversi base64 ke image array
    try:
        # Hapus header base64
        header, encoded = image_data.split(",", 1)
        binary_data = base64.b64decode(encoded)
        img_array = np.frombuffer(binary_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return jsonify({"error": "Gagal memproses gambar"}), 400
    
    if img is None:
        return jsonify({"error": "Gambar tidak valid"}), 400
    
    # Preprocessing dan ekstraksi fitur
    features = preprocess_image(img_array=img)
    if features is None:
        return jsonify({"error": "Gagal mengekstrak fitur gambar"}), 400
    
    # Pilih model (default SVM)
    model_type = data.get('model', 'svm')
    if model_type == 'svm':
        prediction = svm_model.predict(features)
    else:  # knn
        prediction = knn_model.predict(features)
    
    # Decode label
    class_name = le.inverse_transform(prediction)[0]
    
    # Simpan gambar untuk ditampilkan (opsional)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"webcam_{timestamp}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cv2.imwrite(filepath, img)
    
    # Buat URL untuk gambar
    image_url = f"/static/uploads/{urllib.parse.quote(filename)}"
    
    return jsonify({
        "prediction": class_name,
        "model_used": model_type.upper(),
        "image_url": image_url
    })

@app.route('/result')
def show_result():
    # Ambil parameter dari URL
    image_url = request.args.get('image_url', '')
    prediction = request.args.get('prediction', '')
    model_used = request.args.get('model_used', '')
    
    return render_template('result.html', 
                          image_url=image_url,
                          prediction=prediction,
                          model_used=model_used)

if __name__ == '__main__':
    # Buat folder upload jika belum ada
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)