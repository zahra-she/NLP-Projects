"""
IMDB Sentiment Analysis with LSTM
Author: [Your Name]
Description: Train an LSTM model to classify IMDB movie reviews as positive or negative.
"""

import os
import tensorflow as tf
from tensorflow import keras
from keras import layers

# ==============================
# Configurations
# ==============================
BATCH_SIZE = 32
MAX_LENGTH = 600    # Maximum number of tokens per review
MAX_TOKENS = 20000  # Vocabulary size
EPOCHS = 10

# Reduce TensorFlow logging noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ==============================
# 1. Download and extract IMDB dataset
# ==============================
url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset_path = keras.utils.get_file("aclImdb_v1.tar.gz", origin=url, extract=True)
dataset_dir = os.path.join(os.path.dirname(dataset_path), "aclImdb")

# ==============================
# 2. Load train and test datasets
# ==============================
train_ds = keras.utils.text_dataset_from_directory(
    os.path.join(dataset_dir, "train"),
    batch_size=BATCH_SIZE
)

test_ds = keras.utils.text_dataset_from_directory(
    os.path.join(dataset_dir, "test"),
    batch_size=BATCH_SIZE
)

# ==============================
# 3. Prepare text vectorization layer
# ==============================
text_vectorization = layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode="int",
    output_sequence_length=MAX_LENGTH
)

# Extract only text for vocabulary adaptation
text_only_train_ds = train_ds.map(lambda x, y: x)
text_vectorization.adapt(text_only_train_ds)

# ==============================
# 4. Convert text to integer sequences
# ==============================
int_train_ds = train_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
int_test_ds = test_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)

# Prefetch to improve speed
int_train_ds = int_train_ds.prefetch(tf.data.AUTOTUNE)
int_test_ds = int_test_ds.prefetch(tf.data.AUTOTUNE)

# ==============================
# 5. Build the LSTM model
# ==============================
inputs = keras.Input(shape=(None,), dtype="int64")
embedded = layers.Embedding(input_dim=MAX_TOKENS, output_dim=256)(inputs)
x = layers.LSTM(32)(embedded)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.summary()

# ==============================
# 6. Train the model
# ==============================
model.fit(
    int_train_ds,
    validation_data=int_test_ds,
    epochs=EPOCHS
)

# ==============================
# 7. Save the model
# ==============================
model.save("imdb_lstm_model")
print("✅ Model saved as 'imdb_lstm_model/'")
