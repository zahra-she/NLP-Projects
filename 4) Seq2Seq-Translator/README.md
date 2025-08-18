# Seq2Seq-Translator

**Description:**  
This project implements a Neural Machine Translation system to translate English sentences into Spanish using a **Seq2Seq model with GRU layers** in TensorFlow/Keras.

---

## Features
- Encoder-Decoder architecture with **Bidirectional GRU** encoder
- GRU decoder with teacher forcing
- Handles variable-length sentences using padding
- Custom text standardization for Spanish
- Training & validation metrics visualization
- **Ready-to-use inference function** for translating new sentences

---

## Installation

```bash
git clone https://github.com/yourusername/Seq2Seq-Translator.git
cd Seq2Seq-Translator
pip install -r requirements.txt

## Requirements:
Python 3.8+
TensorFlow 2.x
Matplotlib

## Dataset:
Example: Tatoeba English-Spanish sentences
Format: english_sentence \t spanish_sentence per line
Place the dataset file as spa.txt in the project folder.

## Usage
1. Train the model : 
python seq2seq_translator.py

2. Translate a new sentence :
from seq2seq_translator import translate_sentence, seq2seq_rnn, source_vectorizer, target_vectorizer
english_sentence = "I love machine learning"
spanish_translation = translate_sentence(seq2seq_rnn, english_sentence, source_vectorizer, target_vectorizer)
print("English:", english_sentence)
print("Spanish:", spanish_translation)

## Example Output:
English: I love machine learning
Spanish: me encanta el aprendizaje automático