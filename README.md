# Alphabet Recognition using CNN
This project is developed using Python and deep learning libraries like Pandas, NumPy, Keras, and TensorFlow. The goal of this project is to recognize the alphabets using Convolutional Neural Networks (CNNs). We have trained the model using the EMNIST dataset and evaluated the model on the same dataset.

## Dataset
The EMNIST dataset contains over 800,000 images of handwritten characters. The dataset has 26 uppercase letters, 26 lowercase letters, and 10 digits. The images are grayscale and have a resolution of 28x28 pixels. We have used the dataset for training and testing the model.

## Model Architecture
The CNN architecture used in this project includes the following layers:

 * Preprocessing: Image/pixel normalization
 * Preprocessing: ImageDataGenerator - Set zoom range, image rotation, shear range
 * Feature Extraction Layer: Two layers of convolution (Conv2D) and pooling (MaxPool2D) with 32 filters
 * Feature Extraction layer is then connected with fully connected dense layer with Dropout, Early stopping, and Learning rate logic implemented to generalize the model

## UI Interfaces
Two UI interfaces have been developed using Gradio for this project, allowing the user to input an image and predict the output.

1. The first interface allows the user to input an RGB image, which is then converted to grayscale, and the prediction is performed on this transformed image.
2. The second interface is a sketchpad input that allows the user to input a handwritten character and perform the prediction.

## Results
The model has achieved an accuracy of 92% on the EMNIST dataset. This means that the model is able to correctly recognize 92% of the alphabets in the dataset. 

## Requirements
Python 3.x
Pandas
NumPy
Keras
TensorFlow
Gradio
