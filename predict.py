"""
predict.py — Classify a single image from the command line.

Usage:
    python predict.py path/to/image.jpg
"""

import sys
import os
import numpy as np
from PIL import Image
import tensorflow as tf

CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

BINS = {
    'cardboard': '📦  Recycling Bin',
    'glass':     '🍶  Glass Recycling',
    'metal':     '🥫  Metal Recycling',
    'paper':     '📄  Paper Recycling',
    'plastic':   '🧴  Plastic Recycling',
    'trash':     '🗑️  General Waste',
}

def predict(image_path):
    if not os.path.exists('model/trash_classifier.h5'):
        print("❌ No model found. Run: python train.py --data_dir <path>")
        sys.exit(1)

    model = tf.keras.models.load_model('model/trash_classifier.h5')

    img = Image.open(image_path).convert('RGB').resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]
    idx = np.argmax(preds)
    label = CLASS_NAMES[idx]

    print(f"\n{'─'*40}")
    print(f"  Image    : {os.path.basename(image_path)}")
    print(f"  Result   : {label.upper()}  ({preds[idx]*100:.1f}%)")
    print(f"  Dispose  : {BINS[label]}")
    print(f"{'─'*40}")
    print("\n  All probabilities:")
    for i, name in enumerate(CLASS_NAMES):
        bar = '█' * int(preds[i] * 30)
        print(f"  {name:10s}  {bar:<30}  {preds[i]*100:5.1f}%")
    print()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    predict(sys.argv[1])
