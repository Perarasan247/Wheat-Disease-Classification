Project: Wheat Disease Classification (CNN)

Task: Classify wheat leaf images into disease categories (Brown rust, Healthy, Loose Smut, Yellow rust)
Reported accuracy: 96% (on validation/test set used during development)

This repository contains:

Training code (TensorFlow / Keras) to build and train a CNN

A saved Keras model for inference (best_wheat_model.h5)

A Flask backend that exposes a /predict endpoint

A simple, modern frontend (HTML/CSS/JavaScript) to upload an image and get a prediction
