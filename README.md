
# Emotion Detection with DistilBERT

This repository contains a Python script to train a DistilBERT model for emotion detection using the GoEmotions dataset. The script processes the dataset, oversamples minority classes, tokenizes the data, trains the model, and evaluates its performance.

## Features
- Loads the GoEmotions dataset from Hugging Face.
- Filters for 10 emotions: anger, sadness, joy, disgust, fear, surprise, gratitude, remorse, curiosity, neutral.
- Uses RandomOverSampler to balance the training data.
- Trains a DistilBERT model with TensorFlow.
- Saves the trained model and tokenizer to a specified directory.

## Requirements
- Python 3.8+
- Libraries listed in `requirements.txt`
- Optional: GPU for faster training (requires TensorFlow with CUDA support)

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/emotion-detection.git
   cd emotion-detection
