import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging

import cv2
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import *
from keras.models import Sequential
from keras.utils import to_categorical

# Set seed for reproducibility
seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

# Dataset and model settings
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20
DATASET_DIR = "UCF50"
CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]

def frames_extraction(video_path):
    """
    Extract frames from a video, resize and normalize them.
    Args:
        video_path: path to video file
    Returns:
        frames_list: list of normalized frames
    """
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)

    for _ in range(SEQUENCE_LENGTH):
        success, frame = video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255.0
        frames_list.append(normalized_frame)

    video_reader.release()
    return frames_list

def create_dataset():
    """
    Create dataset from selected classes.
    Returns:
        features: video frames
        labels: class indices
    """
    features, labels = []

    for class_index, class_name in enumerate(CLASSES_LIST):
        print(f'Extracting data for class: {class_name}')
        files_list = os.listdir(os.path.join(DATASET_DIR, class_name))

        for file_name in files_list:
            video_path = os.path.join(DATASET_DIR, class_name, file_name)
            frames = frames_extraction(video_path)

            # Only include videos with exact SEQUENCE_LENGTH frames
            if len(frames) == SEQUENCE_LENGTH:
                features.append(frames)
                labels.append(class_index)

    return np.array(features), np.array(labels)

# Create dataset
features, labels = create_dataset()
one_hot_labels = to_categorical(labels)

# Train-test split
features_train, features_test, labels_train, labels_test = train_test_split(
    features, one_hot_labels, test_size=0.25, shuffle=True, random_state=seed_constant
)

def create_LRCN_model():
    """
    Build a ConvLSTM (LRCN) model using TimeDistributed CNN + LSTM
    Returns:
        model: compiled Keras model
    """
    model = Sequential()
    
    # Convolutional layers to extract spatial features
    model.add(TimeDistributed(Conv2D(16, (3,3), padding='same', activation='relu'),
                              input_shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(TimeDistributed(MaxPooling2D((4,4))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(32, (3,3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((4,4))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(64, (3,3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2,2))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(64, (3,3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2,2))))
    model.add(TimeDistributed(Dropout(0.25)))

    # Flatten features and pass to LSTM for temporal modeling
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(32))

    # Output layer with softmax
    model.add(Dense(len(CLASSES_LIST), activation='softmax'))

    model.summary()
    return model

# Create and train model
convlstm_model = create_LRCN_model()
convlstm_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

history = convlstm_model.fit(
    x=features_train, y=labels_train,
    epochs=50, batch_size=4, shuffle=True,
    validation_split=0.2
)

# Evaluate model
loss, accuracy = convlstm_model.evaluate(features_test, labels_test)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

# Save model
convlstm_model.save("cnn_lstm_classification.h5")

def plot_metric(history, metric_train, metric_val, title):
    """
    Plot training and validation metrics
    """
    plt.plot(history.history[metric_train], 'blue', label=metric_train)
    plt.plot(history.history[metric_val], 'red', label=metric_val)
    plt.title(title)
    plt.legend()
    plt.show()

# Plot training results
plot_metric(history, 'loss', 'val_loss', 'Loss vs Validation Loss')
plot_metric(history, 'accuracy', 'val_accuracy', 'Accuracy vs Validation Accuracy')
