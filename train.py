"""
train.py — Train the TrashNet classifier using ResNet50V2 transfer learning.

Usage:
    python train.py --data_dir path/to/dataset-resized
    python train.py --data_dir path/to/dataset-resized --fine_tune

Dataset:
    Download from: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
    Or: https://github.com/garythung/trashnet (dataset-resized folder)

    Expected folder structure:
    dataset-resized/
    ├── cardboard/
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── trash/
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25


def build_model(num_classes=6):
    base = ResNet50V2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False  # Freeze base initially

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, base


def fine_tune_model(model, base, train_gen, val_gen, initial_epochs):
    """Unfreeze top layers for fine-tuning."""
    base.trainable = True

    # Freeze all but the last 40 layers
    for layer in base.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Very low LR for fine tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    cb = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint('model/trash_classifier.h5', save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(factor=0.3, patience=3, verbose=1)
    ]

    history_fine = model.fit(
        train_gen,
        epochs=initial_epochs + 15,
        initial_epoch=initial_epochs,
        validation_data=val_gen,
        callbacks=cb
    )
    return history_fine


def plot_history(history, history_fine=None, save_path='model/training_plot.png'):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    if history_fine:
        acc += history_fine.history['accuracy']
        val_acc += history_fine.history['val_accuracy']
        loss += history_fine.history['loss']
        val_loss += history_fine.history['val_loss']

    epochs_range = range(len(acc))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('TrashNet ResNet50V2 Training', fontsize=14)

    ax1.plot(epochs_range, acc, label='Train Accuracy')
    ax1.plot(epochs_range, val_acc, label='Val Accuracy')
    ax1.axvline(x=EPOCHS, color='r', linestyle='--', label='Fine-tune start')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.legend()

    ax2.plot(epochs_range, loss, label='Train Loss')
    ax2.plot(epochs_range, val_loss, label='Val Loss')
    ax2.axvline(x=EPOCHS, color='r', linestyle='--', label='Fine-tune start')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epochs')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n📊 Training plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train TrashNet Classifier with ResNet50V2')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset-resized folder')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Training epochs (phase 1)')
    parser.add_argument('--fine_tune', action='store_true', help='Enable fine-tuning phase')
    args = parser.parse_args()

    # Validate data directory
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        return

    os.makedirs('model', exist_ok=True)

    # Stronger data augmentation for better generalization
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.25,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.2
    )

    train_gen = train_datagen.flow_from_directory(
        args.data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_gen = train_datagen.flow_from_directory(
        args.data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    print(f"\n📂 Classes found: {list(train_gen.class_indices.keys())}")
    print(f"📸 Training samples: {train_gen.samples}")
    print(f"📸 Validation samples: {val_gen.samples}\n")

    model, base = build_model(num_classes=len(train_gen.class_indices))

    model.summary()

    cb = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint('model/trash_classifier.h5', save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(factor=0.3, patience=3, verbose=1)
    ]

    print("\n🚀 Phase 1: Transfer learning with ResNet50V2 (frozen base)...")
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=cb
    )

    history_fine = None
    if args.fine_tune:
        print("\n🔧 Phase 2: Fine-tuning top 40 layers of ResNet50V2...")
        history_fine = fine_tune_model(model, base, train_gen, val_gen, len(history.history['accuracy']))

    # Final evaluation
    print("\n📈 Evaluating on validation set...")
    val_gen.reset()
    loss, acc = model.evaluate(val_gen)
    print(f"\n✅ Final Validation Accuracy: {acc*100:.2f}%")
    print(f"✅ Previous MobileNetV2 Accuracy: 77.80%")
    print(f"✅ Improvement: +{(acc*100 - 77.80):.2f}%")
    print(f"✅ Model saved to: model/trash_classifier.h5")

    plot_history(history, history_fine)


if __name__ == '__main__':
    main()
