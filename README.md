# Plant Seedling Classification using CNN

## Project Overview
This project aims to use deep learning to classify plant seedlings. It's part of an initiative to modernize agriculture using AI and Deep Learning, reducing manual labor in plant sorting and recognition.


## Objective:
The aim of this project is to use a deep learning model to classify plant seedlings through supervised learning.

## Data Description:
The Aarhus University Signal Processing group, in collaboration with the University of Southern Denmark, has recently released a dataset containing images of unique plants belonging to 12 different species at several growth stages.

You are provided with a dataset of images of plant seedlings at various stages of growth.

- Each image has a filename that is its unique id.
- The dataset comprises 12 plant species.
- The goal of the project is to create a classifier capable of determining a plant's species from a photo.

### List of Species

- Black-grass
- Charlock
- Cleavers
- Common Chickweed
- Common Wheat
- Fat Hen
- Loose Silky-bent
- Maize
- Scentless Mayweed
- Shepherds Purse
- Small-flowered Cranesbill
- Sugar beet

****Learning Outcomes****
 
- Pre-processing of image data
- Visualization of images
- Building the CNN
- Evaluating the Model


## Model Files

We have included model files trained on TensorFlow and PyTorch:

- `Plant_Seedling_Model.h5`: TensorFlow model
- `Plant_Seedling_Model.pth`: PyTorch model
- `Plant_Seedling_Model1.h5`: Another TensorFlow model
- `Plant_Seedling_Model1.pth`: Another PyTorch model

## Running the Project

You can run the project directly from the command terminal using the provided Python script:

- `Plant_Seedlings_Classification.py`: This script allows you to classify plant seedlings using the trained model. You can run it with the following command:

  ```bash
  python Plant_Seedlings_Classification.py

## Hosting the Model using Gradio

We have also provided an example of how to host the model using Gradio in the `Plant_Seedling_Classification_Gradio.ipynb` notebook. Gradio allows you to create a simple and user-friendly web interface to interact with your trained model.

## Installation
To set up your environment to run this code, you will need to install the necessary Python libraries listed in `requirements.txt`. You can do this by running the following command:

```bash
pip install -r requirements.txt
