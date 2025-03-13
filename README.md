# RNN-LSTM-Image-Captioning-in-PyTorch
An RNN-LSTM model for image captioning using PyTorch, mapping CNN-extracted features to text. Includes implementation from scratch, model training, and evaluation using BLEU scores.

## Overview
This project is part of the Deep Learning course at Reichman University. It focuses on **Recurrent Neural Networks (RNNs)** and **Long Short-Term Memory (LSTM) networks** for sequence modeling and image captioning.

This project implements an **RNN-based model** using **PyTorch** to generate captions for images. The model learns to map image features to sequences of words, using LSTMs to handle sequential dependencies.

The project is divided into three main parts:
- **Implementing RNN and LSTM forward and backward propagation from scratch.**
- **Training an image captioning model using CNN features and an LSTM decoder.**
- **Generating and evaluating image captions using a trained model.**

## Dataset
The dataset used for image captioning is derived from **MS COCO (Microsoft Common Objects in Context)**. The dataset consists of images paired with multiple human-written captions.

## Project Structure
### Data Loading & Preprocessing
- Loads a dataset of images and their captions.
- Extracts CNN-based image features.
- Tokenizes and encodes text captions.

### RNN & LSTM Implementation
- Implements a basic RNN forward and backward pass.
- Extends to LSTM cells for better sequence modeling.
- Uses an embedding layer to map words to vector representations.

### Model Training & Evaluation
- Trains the LSTM-based captioning model using cross-entropy loss.
- Evaluates performance using BLEU scores for generated captions.
- Visualizes generated captions for test images.

## Installation & Requirements
To run this notebook, you need **Python 3** and the following dependencies:

```sh
pip install torch torchvision numpy matplotlib nltk
```

## Usage
Clone the repository:

```sh
git clone https://github.com/LidanAvisar/RNN-LSTM-Image-Captioning-in-PyTorch
cd RNN-LSTM-Image-Captioning-in-PyTorch
```

Open the Jupyter Notebook:

```sh
jupyter notebook "RNN + LSTM network + Image Captioning in PyTorch.ipynb"
```

Run all cells to train and evaluate the captioning model.

## Results & Analysis
The notebook visualizes:
- Sample images and their corresponding generated captions.
- RNN and LSTM learning curves over training epochs.
- BLEU score analysis for caption quality.
