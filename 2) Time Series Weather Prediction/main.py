import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers

# -------------------------
# 1. Load Data
# -------------------------
data = pd.read_csv("jena_climate_2009_2016.csv")

# Use all columns except the first (timestamp) as features
raw_data = data.iloc[:, 1:]
# Temperature column is the target
temperature = data.iloc[:, 2]

# -------------------------
# 2. Split Data
# -------------------------
num_train_samples = int(0.5 * len(raw_data))
num_val_samples = int(0.25 * len(raw_data))
num_test_samples = len(raw_data) - num_train_samples - num_val_samples

print("Train samples:", num_train_samples)
print("Validation samples:", num_val_samples)
print("Test samples:", num_test_samples)

# -------------------------
# 3. Normalize Data
# -------------------------
mean = raw_data[:num_train_samples].mean(axis=0)
std = raw_data[:num_train_samples].std(axis=0)
raw_data -= mean
raw_data /= std

# -------------------------
# 4. Prepare Time Series Dataset
# -------------------------
sampling_rate = 6          # Take one data point every 6 minutes
sequence_length = 120      # Look at previous 5 days
delay = sampling_rate * (sequence_length + 24 - 1)  # Predict 24 hours later
batch_size = 256

train_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0,
    end_index=num_train_samples
)

val_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples
)

test_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples + num_val_samples
)

# Check shapes
for samples, targets in train_dataset.take(1):
    print("Sample batch shape:", samples.shape)  # (batch, seq_len, features)
    print("Target batch shape:", targets.shape)
    break

# -------------------------
# 5. Build Models
# -------------------------
def build_lstm_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.LSTM(32, recurrent_dropout=0.25)(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    return model

def build_gru_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.GRU(32, recurrent_dropout=0.5, return_sequences=True)(inputs)
    x = layers.GRU(32, recurrent_dropout=0.5)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    return model

def build_bidirectional_lstm_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Bidirectional(layers.LSTM(16))(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    return model

# -------------------------
# 6. Train Models
# -------------------------
models = {
    "LSTM": build_lstm_model((sequence_length, raw_data.shape[-1])),
    "GRU": build_gru_model((sequence_length, raw_data.shape[-1])),
    "Bidirectional_LSTM": build_bidirectional_lstm_model((sequence_length, raw_data.shape[-1]))
}

histories = {}

for name, model in models.items():
    print(f"\nTraining {name} model...")
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=10)
    histories[name] = history

# -------------------------
# 7. Plot Training & Validation MAE
# -------------------------
plt.figure(figsize=(10, 6))
for name, history in histories.items():
    plt.plot(history.history["mae"], label=f"{name} Train MAE")
    plt.plot(history.history["val_mae"], label=f"{name} Val MAE")

plt.title("Training and Validation MAE")
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.legend()
plt.show()