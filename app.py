import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

DISPOSAL_TIPS = {
    'cardboard': {
        'emoji': '📦',
        'bin': 'Recycling Bin',
        'color': '#8B6914',
        'tip': 'Flatten boxes before recycling. Remove any tape or staples if possible.'
    },
    'glass': {
        'emoji': '🍶',
        'bin': 'Glass Recycling',
        'color': '#4A7C59',
        'tip': 'Rinse the container. Do not mix with broken glass — dispose separately.'
    },
    'metal': {
        'emoji': '🥫',
        'bin': 'Metal Recycling',
        'color': '#6B7280',
        'tip': 'Rinse cans. Aluminium and steel are both recyclable. Crush to save space.'
    },
    'paper': {
        'emoji': '📄',
        'bin': 'Paper Recycling',
        'color': '#3B82F6',
        'tip': 'Keep dry. Shredded paper goes in a tied bag. Avoid greasy or wax-coated paper.'
    },
    'plastic': {
        'emoji': '🧴',
        'bin': 'Plastic Recycling',
        'color': '#EF4444',
        'tip': 'Check the recycling number. Rinse containers. Remove caps if different material.'
    },
    'trash': {
        'emoji': '🗑️',
        'bin': 'General Waste',
        'color': '#374151',
        'tip': 'This item cannot be recycled. Dispose in general waste bin responsibly.'
    }
}

# Load model once at startup
model = None

def load_model():
    global model
    model_path = 'model/trash_classifier.h5'
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded from disk.")
    else:
        print("⚠️  No trained model found. Run train.py first to train the model.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please run train.py first.'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use PNG, JPG, or JPEG.'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        img_array = preprocess_image(filepath)
        predictions = model.predict(img_array)[0]
        predicted_idx = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(predictions[predicted_idx]) * 100

        all_probs = {
            CLASS_NAMES[i]: round(float(predictions[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        result = {
            'class': predicted_class,
            'confidence': round(confidence, 2),
            'all_probabilities': all_probs,
            'disposal': DISPOSAL_TIPS[predicted_class]
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    load_model()
    print("\n🌐 Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000)
