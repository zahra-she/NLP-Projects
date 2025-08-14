# 🎬 IMDB Sentiment Analysis with LSTM

This project trains an **LSTM-based neural network** to classify IMDB movie reviews as **positive** or **negative** using TensorFlow and Keras.

## 📌 Features
- Automatic download of the IMDB dataset.
- Text preprocessing with **TextVectorization**.
- LSTM layer for sequence modeling.
- Saves trained model for future inference.

## 📂 Dataset
We use the [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/), which contains:
- **25,000 training reviews**
- **25,000 test reviews**
- Balanced positive and negative samples.

The dataset is automatically downloaded by `train.py`.

## 🚀 How to Run
1. Clone the repository:
git clone https://github.com/yourusername/imdb-sentiment-lstm.git
cd imdb-sentiment-lstm

2. Train and Save Model

After training, the model is saved in:
imdb_lstm_model/

3. You can load it with:

from tensorflow import keras
model = keras.models.load_model("imdb_lstm_model")

📊 Expected Results:
Accuracy: 85–90% after ~10 epochs.

🧠 Model Architecture
Embedding (20000 vocab, 256 dims) → LSTM(32) → Dense(1, sigmoid)