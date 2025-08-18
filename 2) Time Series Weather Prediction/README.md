# Time-Series Weather Prediction

A hands-on project for predicting temperature using deep learning models on time series data. This project uses the [Jena Climate dataset](https://www.bgc-jena.mpg.de/wetter/) and implements LSTM, GRU, and Bidirectional LSTM models.

## Features
- Data normalization and preprocessing
- Time series dataset creation with Keras
- LSTM-based regression model
- Optional GRU and Bidirectional LSTM models for comparison
- Training and validation visualization (MAE)

## Dataset
[Jena Climate 2009-2016 Dataset](https://www.bgc-jena.mpg.de/wetter/)  
- 14 climate features (temperature, pressure, humidity, wind, etc.)
- 10-minute interval measurements
- We use temperature as target for prediction

## Installation
```bash
pip install tensorflow pandas matplotlib

## Usage

Download jena_climate_2009_2016.csv dataset
Run weather_prediction_timeseries.py to train and evaluate the models
Check MAE plots for training and validation performance

## Models Implemented

LSTM
GRU
Bidirectional LSTM

## Notes

sequence_length = 120 (previous 5 days)
sampling_rate = 6 (every 6 steps)
delay = 24 hours ahead prediction
Dropout layers are used to reduce overfitting