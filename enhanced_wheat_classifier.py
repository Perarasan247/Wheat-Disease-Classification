#!/usr/bin/env python3
"""
Enhanced Wheat Disease Classification with CNN
Addresses issues found in original notebook and implements improvements
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
CONFIG = {
    'data_dir': r'/kaggle/input/wheat-leaf-disease/Dataset',
    'img_size': (224, 224),  # Increased from 150x150
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'seed': 42
}

def create_data_generators():
    """Enhanced data augmentation with class balancing"""
    
    # More aggressive augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        channel_shift_range=20,
        fill_mode='nearest',
        validation_split=CONFIG['validation_split']
    )
    
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=CONFIG['validation_split']
    )
    
    train_generator = train_datagen.flow_from_directory(
        CONFIG['data_dir'],
        target_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='training',
        seed=CONFIG['seed']
    )
    
    val_generator = val_datagen.flow_from_directory(
        CONFIG['data_dir'],
        target_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=CONFIG['seed']
    )
    
    # Debug: Check data generator output
    print(f"Train generator classes: {train_generator.num_classes}")
    print(f"Class indices: {train_generator.class_indices}")
    
    if train_generator.num_classes < 2:
        raise ValueError(f"Only {train_generator.num_classes} class found. Check your data directory structure. Expected 4 classes: Brown rust, Healthy, Loose Smut, Yellow rust")
    
    return train_generator, val_generator

def create_enhanced_model(num_classes=4):
    """Enhanced CNN with transfer learning base"""
    
    # Use EfficientNetB0 as base (lightweight but effective)
    base_model = tf.keras.applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*CONFIG['img_size'], 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax', name='predictions')
    ])
    
    return model

def create_callbacks():
    """Enhanced callbacks for better training"""
    
    return [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            'best_wheat_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

def calculate_class_weights(train_generator):
    """Calculate class weights to handle imbalance"""
    
    # Get class distribution
    class_counts = np.bincount(train_generator.classes)
    total_samples = sum(class_counts)
    
    # Calculate weights
    class_weights = {}
    for i, count in enumerate(class_counts):
        class_weights[i] = total_samples / (len(class_counts) * count)
    
    print("Class distribution:", dict(zip(train_generator.class_indices.keys(), class_counts)))
    print("Class weights:", class_weights)
    
    return class_weights

def plot_training_history(history):
    """Plot training metrics"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, val_generator):
    """Comprehensive model evaluation"""
    
    # Predictions
    val_generator.reset()
    predictions = model.predict(val_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes
    
    # Classification report
    class_names = list(val_generator.class_indices.keys())
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate accuracy
    accuracy = np.sum(y_pred == y_true) / len(y_true)
    print(f"\nFinal Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy, predictions

def fine_tune_model(model, train_generator, val_generator, class_weights):
    """Fine-tune the pre-trained model"""
    
    print("Starting fine-tuning...")
    
    # Unfreeze top layers of base model
    model.layers[0].trainable = True
    
    # Use lower learning rate for fine-tuning
    model.compile(
        optimizer=optimizers.Adam(learning_rate=CONFIG['learning_rate']/10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tune with fewer epochs
    fine_tune_callbacks = [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint('fine_tuned_wheat_model.h5', monitor='val_accuracy', save_best_only=True)
    ]
    
    history_fine = model.fit(
        train_generator,
        epochs=20,
        validation_data=val_generator,
        class_weight=class_weights,
        callbacks=fine_tune_callbacks,
        verbose=1
    )
    
    return history_fine

def main():
    """Main training pipeline"""
    
    print("Enhanced Wheat Disease Classification")
    print("=" * 50)
    
    # Set random seeds
    tf.random.set_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    # Create data generators
    print("Creating data generators...")
    train_gen, val_gen = create_data_generators()
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_gen)
    
    # Create model
    print("Creating enhanced model...")
    num_classes = len(train_gen.class_indices)
    print(f"Number of classes detected: {num_classes}")
    model = create_enhanced_model(num_classes)
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Model summary:")
    model.summary()
    
    # Initial training
    print("Starting initial training...")
    history = model.fit(
        train_gen,
        epochs=CONFIG['epochs'],
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=create_callbacks(),
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    print("Evaluating model...")
    accuracy, predictions = evaluate_model(model, val_gen)
    
    # Fine-tuning (optional)
    if accuracy < 0.85:  # Only fine-tune if accuracy is below 85%
        print("Accuracy below 85%, starting fine-tuning...")
        history_fine = fine_tune_model(model, train_gen, val_gen, class_weights)
        
        # Re-evaluate after fine-tuning
        print("Re-evaluating after fine-tuning...")
        accuracy, predictions = evaluate_model(model, val_gen)
    
    print(f"\nTraining completed! Final accuracy: {accuracy*100:.2f}%")
    
    # Save final model
    model.save('final_enhanced_wheat_model.h5')
    model.save('best_wheat_model.h5')  # Save as app expects
    print("Model saved as 'final_enhanced_wheat_model.h5' and 'best_wheat_model.h5'")

if __name__ == "__main__":
    main()