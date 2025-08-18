# Neural Machine Translation (English → Spanish) using RNN-based Seq2Seq model with GRU layers in TensorFlow/Keras.

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import re
import string
import random
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt

# ===========================
# Hyperparameters
# ===========================
BATCH_SIZE = 64
VOCAB_SIZE = 15000
SEQ_LENGTH = 20
EMBED_DIM = 256
LATENT_DIM = 1024

# ===========================
# Load Data
# ===========================
def load_document(path):
    """Load the dataset from a text file and return list of lines."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")[:-1]
    return lines

def create_text_pairs(lines):
    """
    Convert dataset lines into English-Spanish pairs.
    Adds [start] and [end] tokens to Spanish sentences.
    """
    text_pairs = []
    for line in lines:
        english, spanish = line.split("\t")
        spanish = "[start] " + spanish + " [end]"
        text_pairs.append((english, spanish))
    print("[INFO] Sample data:", random.choice(text_pairs))
    return text_pairs

def split_dataset(pairs, val_ratio=0.15):
    """
    Shuffle and split dataset into training, validation, and test sets.
    """
    random.shuffle(pairs)
    num_val = int(val_ratio * len(pairs))
    num_train = len(pairs) - 2 * num_val
    train_pairs = pairs[:num_train]
    val_pairs = pairs[num_train:num_train+num_val]
    test_pairs = pairs[num_train+num_val:]
    print("[INFO] Dataset split completed.")
    return train_pairs, val_pairs, test_pairs

# ===========================
# Text Standardization & Tokenization
# ===========================
def custom_standardization(input_string):
    """
    Lowercase text and remove punctuation except [start] and [end].
    """
    strip_chars = string.punctuation + "¿"
    strip_chars = strip_chars.replace("[", "")
    strip_chars = strip_chars.replace("]", "")
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, f"[{re.escape(strip_chars)}]", "")

def tokenize_dataset(train_pairs):
    """
    Create TextVectorization layers for English (source) and Spanish (target).
    """
    source_vectorizer = layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=SEQ_LENGTH
    )
    target_vectorizer = layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=SEQ_LENGTH + 1,
        standardize=custom_standardization
    )
    # Fit vectorizers on training text
    source_vectorizer.adapt([p[0] for p in train_pairs])
    target_vectorizer.adapt([p[1] for p in train_pairs])
    print("[INFO] Data tokenization complete.")
    return source_vectorizer, target_vectorizer

# ===========================
# Dataset Preparation
# ===========================
def format_dataset(eng, spa):
    """
    Prepare input-output pairs for training:
    - English sentence → encoder input
    - Spanish sentence shifted → decoder input and target
    """
    eng = source_vectorizer(eng)
    spa = target_vectorizer(spa)
    return (
        {
            "english": eng,
            "spanish": spa[:, :-1]  # decoder input
        },
        spa[:, 1:]  # target output
    )

def make_dataset(pairs):
    """
    Convert list of sentence pairs into tf.data.Dataset ready for training.
    """
    eng_texts, spa_texts = zip(*pairs)
    dataset = tf.data.Dataset.from_tensor_slices((list(eng_texts), list(spa_texts)))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(format_dataset, num_parallel_calls=4)
    return dataset.shuffle(2048)

# ===========================
# Model Definition
# ===========================
def define_seq2seq_model():
    """
    Create a Seq2Seq model using Bidirectional GRU encoder and GRU decoder.
    """
    # Encoder input
    source_input = layers.Input(shape=(None,), dtype="int64", name="english")
    x = layers.Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True)(source_input)

    # Decoder input
    target_input = layers.Input(shape=(None,), dtype="int64", name="spanish")
    y = layers.Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True)(target_input)

    # Encoder: Bidirectional GRU
    encoded = layers.Bidirectional(layers.GRU(LATENT_DIM), merge_mode="sum")(x)

    # Decoder GRU
    y = layers.GRU(LATENT_DIM, return_sequences=True)(y, initial_state=encoded)
    y = layers.TimeDistributed(layers.Dropout(0.5))(y)
    output = layers.TimeDistributed(layers.Dense(VOCAB_SIZE, activation="softmax"))(y)

    return keras.Model([source_input, target_input], output)

# ===========================
# Training Metrics Visualization
# ===========================
def plot_metric(history, metric_train, metric_val, title):
    """Plot training vs validation metric over epochs."""
    train_values = history.history[metric_train]
    val_values = history.history[metric_val]
    epochs = range(len(train_values))
    plt.plot(epochs, train_values, 'blue', label=metric_train)
    plt.plot(epochs, val_values, 'red', label=metric_val)
    plt.title(title)
    plt.legend()
    plt.show()

# ===========================
# MAIN
# ===========================
if __name__ == "__main__":
    # Load and prepare data
    lines = load_document("spa.txt")
    pairs = create_text_pairs(lines)
    train_pairs, val_pairs, test_pairs = split_dataset(pairs)
    source_vectorizer, target_vectorizer = tokenize_dataset(train_pairs)

    # Create datasets
    train_ds = make_dataset(train_pairs)
    val_ds = make_dataset(val_pairs)

    # Check shapes
    for inputs, targets in train_ds.take(1):
        print(f"inputs['english'].shape: {inputs['english'].shape}")
        print(f"inputs['spanish'].shape: {inputs['spanish'].shape}")
        print(f"targets.shape: {targets.shape}")

    # Define model
    seq2seq_rnn = define_seq2seq_model()
    seq2seq_rnn.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train model
    history = seq2seq_rnn.fit(train_ds, epochs=15, validation_data=val_ds)

    # Plot training metrics
    plot_metric(history, 'loss', 'val_loss', 'Training vs Validation Loss')
    plot_metric(history, 'accuracy', 'val_accuracy', 'Training vs Validation Accuracy')


# ===========================
# Inference Function
# ===========================
def translate_sentence(model, sentence, source_vectorizer, target_vectorizer, max_len=SEQ_LENGTH):
    """
    Translate a single English sentence to Spanish using the trained Seq2Seq model.
    
    Args:
        model: Trained Seq2Seq Keras model
        sentence: English sentence (string)
        source_vectorizer: TextVectorization for English
        target_vectorizer: TextVectorization for Spanish
        max_len: Maximum sequence length for decoder output

    Returns:
        translated_sentence: Predicted Spanish sentence (string)
    """
    # Prepare encoder input
    encoder_input = source_vectorizer([sentence])
    
    # Initialize decoder input with [start] token
    start_token_index = target_vectorizer(['[start]']).numpy()[0][0]
    decoder_input = tf.expand_dims([start_token_index], 0)  # shape (1, 1)
    
    translated_tokens = []

    for _ in range(max_len):
        # Predict next token probabilities
        predictions = model([encoder_input, decoder_input], training=False)
        next_token_id = tf.argmax(predictions[:, -1, :], axis=-1).numpy()[0]
        
        # Stop if [end] token is predicted
        if next_token_id == target_vectorizer(['[end]']).numpy()[0][0]:
            break

        translated_tokens.append(next_token_id)
        
        # Append predicted token to decoder input
        decoder_input = tf.concat([decoder_input, tf.expand_dims([next_token_id], 0)], axis=1)
    
    # Convert tokens to words
    index_to_word = dict((i, w) for i, w in enumerate(target_vectorizer.get_vocabulary()))
    translated_words = [index_to_word.get(token, '') for token in translated_tokens]
    
    return ' '.join(translated_words)
