Video Action Recognition with ConvLSTM (LRCN)

This repository contains a video action recognition project using a ConvLSTM (LRCN) model implemented in Keras and TensorFlow. The model extracts spatial features from video frames with a CNN and captures temporal dynamics using LSTM.
-------------------------------------------------------

## Dataset

This project uses the UCF50 dataset. Only a subset of classes is used for demonstration:

WalkingWithDog
TaiChi
Swing
HorseRace

Make sure to download the dataset and place it in a folder named UCF50 in the project root.

## Project Structure
UCF50/                # Dataset directory
├── WalkingWithDog/
├── TaiChi/
├── Swing/
└── HorseRace/

cnn_lstm_classification.h5   # Saved trained model
video_action_recognition.py  # Main Python script
README.md                     # This file

## Requirements

Python 3.8+
TensorFlow 2.x
Keras
OpenCV (cv2)
NumPy
Matplotlib
scikit-learn

You can install the requirements using:

pip install tensorflow keras opencv-python numpy matplotlib scikit-learn

## How to Run

1. Clone the repository:

git clone <your-repo-url>
cd <repo-folder>

2. Ensure the dataset is downloaded and placed in the UCF50 folder.

3. Run the main script:

python video_action_recognition.py


This will:

Extract frames from the videos
Create train/test datasets
Train the ConvLSTM model
Save the trained model as cnn_lstm_classification.h5
Plot training/validation metrics

## Model Architecture

TimeDistributed CNN: Extracts spatial features from each frame

LSTM: Captures temporal dependencies between frames

Dense Softmax: Outputs probability for each action class

## Results

After training, the model outputs:

Training and validation loss and accuracy plots
Test set evaluation: Loss and Accuracy

Example:
Test Loss: 0.2345, Test Accuracy: 0.9123

## References

UCF50 Dataset

Donahue, Jeff, et al. “Long-term Recurrent Convolutional Networks for Visual Recognition and Description.” CVPR 2015.