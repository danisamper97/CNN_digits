# CNN Digits Classifier

This project demonstrates how to build, train, and evaluate a **Convolutional Neural Network (CNN)** to classify handwritten digits from the **MNIST dataset**. The project is implemented in a Jupyter Notebook (`cnn_digits.ipynb`) and includes functionality for testing the model on custom handwritten digits.

---

## Overview
The goal of this project is to classify handwritten digits (0–9) using a CNN. The MNIST dataset, which contains 28x28 grayscale images of digits, is used for training and testing. The project also includes functionality to test the model on custom handwritten digit images.

---

## Features
- **Dataset**: Uses the MNIST dataset for training and testing.
- **Model Architectures**:
  - Fully Connected Neural Network (FCNN).
  - Convolutional Neural Network (CNN).
- **Experiment Tracking**: Tracks training runs using **Comet.ml**.
- **Custom Image Testing**: Allows testing the model on user-provided handwritten digit images.
- **Visualization**: Displays random samples from the dataset and classification results.

---

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook
- Libraries: TensorFlow, NumPy, Matplotlib, OpenCV, Comet.ml

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/danisamper97/CNN_digits
   cd CNN_digits

2. Install the required Python packages:    
    pip install -r requirements.txt
   
3. Set up Comet.ml: 
    - Create a Comet.ml account.
    - Generate an API key from your Comet.ml account.
    - Replace the placeholder COMET_API_KEY in the notebook with your API key.

---

## Project Workflow
1. Loading the Dataset
    - The MNIST dataset is loaded using TensorFlow's tf.keras.datasets.mnist module. It contains:
        1. **Training set**: 60,000 images.
        2. **Test set**: 10,000 images.

2. Data Preprocessing
    - Normalization: Pixel values are scaled to the range [0, 1].
    - Reshaping: Images are reshaped to include a channel dimension (28, 28, 1) for compatibility with CNNs.

3. Building the Models
    - Two models are implemented:

        1. **Fully Connected Neural Network (FCNN)**: A simple architecture with dense layers. Used as a baseline model.
        2. **Convolutional Neural Network (CNN)**: Includes convolutional layers, max-pooling layers, and dense layers. Extracts spatial features for better performance on image data.

4. Training the Models
    - The models are trained using the model.fit method.
    - Hyperparameters:
        - **Learning rate**: 0.0001 (FCNN), 0.0005 (CNN).
        - **Batch size**: 64.
        - **Epochs**: 5.
        - **Optimizer**: Adam optimizer.
        - Training runs are tracked using Comet.ml.

5. Evaluating the Models
    - The models are evaluated on the test dataset using model.evaluate.
    - Metrics:
        1. **Loss**: Measures the error in predictions.
        2. **Accuracy**: Measures the percentage of correct predictions.

6. Testing on Custom Images
    - Users can test the CNN on their own handwritten digit images:

---

## Results
1. Fully Connected Neural Network (FCNN):

    - Training Accuracy: ~92%.
    - Test Accuracy: ~91%.

2. Convolutional Neural Network (CNN):

    - Training Accuracy: ~98%.
    - Test Accuracy: ~97%.

The CNN outperforms the FCNN due to its ability to extract spatial features from images.

---

## Ackowledgments
- MNIST Dataset: Yann LeCun's MNIST Database.
- TensorFlow: For providing tools to build and train neural networks.
- Comet.ml: For experiment tracking and visualization.

Feel free to contribute to this project by submitting pull requests or reporting issues!