#!/usr/bin/env python3
"""
Complete CNN Wheat Disease Classification
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
CONFIG = {
    'data_dir': 'Dataset',
    'img_size': (224, 224),
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'seed': 42
}

def create_data_generators():
    """Create data generators with augmentation"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
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
    
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Classes: {list(train_generator.class_indices.keys())}")
    
    return train_generator, val_generator

def create_cnn_model(num_classes=4):
    """Enhanced CNN model"""
    model = models.Sequential([
        # First block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*CONFIG['img_size'], 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth block
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Classifier
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

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
    """Evaluate model performance"""
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
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    accuracy = np.sum(y_pred == y_true) / len(y_true)
    print(f"\nFinal Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy

def main():
    """Complete training pipeline"""
    print("CNN Wheat Disease Classification")
    print("=" * 40)
    
    # Set seeds
    tf.random.set_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    # Create data generators
    print("\nCreating data generators...")
    train_gen, val_gen = create_data_generators()
    
    # Create CNN model
    print("\nCreating CNN model...")
    model = create_cnn_model(len(train_gen.class_indices))
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy', 
            patience=15, 
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
            patience=7,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        train_gen,
        epochs=CONFIG['epochs'],
        validation_data=val_gen,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history)
    
    # Evaluate model
    print("\nEvaluating model...")
    accuracy = evaluate_model(model, val_gen)
    
    # Save final model
    model.save('best_wheat_model.h5')
    print(f"\nTraining completed! Final accuracy: {accuracy*100:.2f}%")
    print("Model saved as 'best_wheat_model.h5'")

if __name__ == "__main__":
    main()