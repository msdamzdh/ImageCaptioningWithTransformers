# 1. Overview
The notebook provides a comprehensive solution for image captioning, which involves feature extraction from images using a Convolutional Neural Network (CNN) and sequence generation using a Transformer-based model. The project assumes that a labeled dataset, containing images and corresponding captions, is available and ready for training.

![Network flowchart](https://github.com/user-attachments/assets/1f7e9db2-294b-441a-89be-aecbf948e361)

# 2. Modules and Implementation
## 2.1 Step 1: Import Necessary Libraries
This section imports the required Python libraries, including:
  -	PyTorch (torch, torch.nn, torchvision): For building and training the neural networks.
  -	Dataset Handling (torch.utils.data): To load and preprocess the dataset.
  -	Computer Vision (torchvision.models, torchvision.transforms, torchvision.io): To work with image data and perform image transformations.
  -	NLP Tools (nltk.tokenize): For tokenizing the captions.
  -	Matplotlib: For displaying images and plotting learning curves.
```
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision.io import read_image
import pandas as pd
from nltk.tokenize import wordpunct_tokenize
```
## 2.2 Step 2: Load Dataset
The custom dataset class CustomImageDataset is defined to handle loading image data and their corresponding captions. It reads captions from a CSV file and tokenizes them using nltk. It also creates a vocabulary and maps words to indices for input to the model.
Key functionality:
  -	Tokenization and Vocabulary Creation: Captions are tokenized, and a vocabulary dictionary maps each word to a unique index.
  -	Padding and Index Conversion: Sentences are converted to a sequence of indices and padded to a fixed length.
```
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        # Load image labels and captions from the CSV file
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        ...
```
Example Usage:
```
# Initialize the dataset
full_dataset = CustomImageDataset(annotations_file='captions.csv', img_dir='images/', transform=your_transform)
```
## 2.3 Step 3: Feature Extraction using CNN (ResNet)
The class CNNEncoder is defined using a pre-trained ResNet-50 model from torchvision. This model is used to extract image features which will be passed as input to the Transformer model for caption generation.
Key functionality:
  -	Feature Extraction: Uses ResNet-50 to extract high-level features.
  -	Embedding Layer: Adds a fully connected layer to reduce feature dimensionality.
```
class CNNEncoder(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        # Load pre-trained ResNet50 and freeze its layers
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        ...
```
## 2.4 Step 4: Caption Generation using Transformer
The class TransformerCaptioningModel is defined using PyTorch’s nn.Transformer module. It implements an encoder-decoder architecture that uses the extracted image features as the encoder's input and the caption sequence as the decoder’s input.
Key functionality:
  -	Embeddings for Captions: Converts word indices into dense vector representations.
  -	Transformer Network: Uses Transformer architecture with multi-head attention and feedforward networks for sequence modeling.
  -	Masking: Implements square subsequent masks to prevent attending to future tokens.
```
class TransformerCaptioningModel(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size, num_layers, max_seq_len, device):
        super().__init__()
        # Define the Transformer model and related layers
        ...
```
## 2.5 Step 5: Training the Model
The function train_model trains the CNN encoder and Transformer decoder using the provided dataset.
Key functionality:
  -	Adam Optimizer: Optimizes both encoder and decoder parameters.
  -	Cross Entropy Loss: Measures how well the generated captions match the ground truth captions.
  -	Loss Tracking: Prints and records the loss for each epoch.
```
def train_model(encoder, decoder, loss_fn, dataloader, num_epochs, learning_rate, vocab_size, device):
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
    ...
```
## 2.6 Step 6: Model Initialization and Dataset Split
Initializes the CNN encoder, Transformer decoder, and loads the dataset using the custom CustomImageDataset class. The dataset is split into training and test sets.
```
embed_size = 256
hidden_size = 512
...
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2])
...
```
## 2.7 Step 7: Plotting Learning Curve
This section plots the training loss over the epochs to visualize the learning progress.
```
fig, ax = plt.subplots()
line_up, = ax.plot(range(1,len(loss_vec)+1),loss_vec, label='Line 1')
ax.legend([line_up], ['Loss'])
```
## 2.8 Step 8: Image Captioning
Given an image, the trained encoder and decoder generate a caption. The caption is displayed below the image.
Key functionality:
  -	Image Preprocessing: Applies transformations to the input image.
  -	Caption Generation: Uses the trained decoder to predict the next word in the sequence until an end token is generated or the maximum length is reached.
```
Image_Address = 'path/to/image.jpg'
image = read_image(Image_Address)
...
title = ' '.join([x for x in token if x not in special_tokens])
plt.title(title)
```
## 3. Summary
This notebook implements an end-to-end image captioning system using a combination of a CNN for image feature extraction and a Transformer model for caption generation. Key steps include dataset preparation, model definition, training, evaluation, and caption generation. This project highlights the use of state-of-the-art deep learning architectures for multimodal problems like image captioning.

