# gender_age_detection
# Age and Gender Prediction from Facial Images

This project implements a multi-output deep learning model to predict **age** (regression) and **gender** (binary classification) from facial images. It uses the UTKFace dataset and is built with TensorFlow and Keras. The project includes scripts for data visualization, model training, and real-time prediction using a webcam.

## Table of Contents

- [Project Structure](#project-structure)
- [Dataset Setup](#dataset-setup)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dependencies](#dependencies)

## Project Structure

The project consists of three main Jupyter Notebooks:

- `dataVisualization.ipynb`: Explores and visualizes the age and gender distribution of the UTKFace dataset.
- `modelTrainig.ipynb`: Contains the complete pipeline for loading data, building the CNN model, training it, and saving the final model artifacts.
- `modelTesting.ipynb`: Uses the trained model to perform real-time age and gender prediction from a live webcam feed.

## Dataset Setup

This project relies on the **UTKFace new** dataset from Kaggle.

1. **Download the Dataset**
   You can download the dataset from the official Kaggle page:
   <https://www.kaggle.com/datasets/jangedoo/utkface-new?resource=download>

2. **Folder Structure**
   After downloading and extracting the contents, you need to set up the folder structure as follows:

   a. In the root directory of this project, create a new folder named `Dataset`.
   b. From the downloaded files, you only need the `UTKFace` folder for this project. Move it inside the `Dataset` folder.

   The final structure should look like this:

   ```text
   Project/
        ├── Dataset/
        │   └── UTKFace/
        ├── Models/
        │   ├── age_gender_model.h5
        │   └── age_gender_model.keras
        ├── dataVisualization.ipynb
        ├── modelTesting.ipynb
        ├── modelTrainig.ipynb
        └── ... other project files
   ```

## Installation

1. **Create and activate a virtual environment:**
   It is highly recommended to use a virtual environment to manage dependencies.

   ```bash
   # Create a virtual environment
   python -m venv venv

   # Activate on Windows
   .\venv\Scripts\activate

   # Activate on macOS/Linux
   source venv/bin/activate
   ```

2. **Install Dependencies:**
   Install all the required packages using pip.

   ```bash
   pip install numpy pandas seaborn tqdm matplotlib tensorflow keras opencv-python jupyter
   ```

## Usage

Follow these steps to run the notebooks in the correct order.

1. **Data Visualization (Optional)**
   To understand the dataset's characteristics, run the cells in `dataVisualization.ipynb`. This will display plots showing the distribution of ages and genders in the dataset.

2. **Train the Model**
   a. Open and run all cells in `modelTrainig.ipynb`.
   b. This notebook will: - Load the images from `Dataset/UTKFace/`. - Pre-process the images (resize to 128x128 grayscale, normalize). - Build and compile the multi-output CNN model. - Train the model for 30 epochs. - Save the trained model as `Models/age_gender_model.h5` and `Models/age_gender_model.keras`.
   c. Ensure a `Models/` directory exists in the project root before running the final cells, or create it if it doesn't exist.

3. **Real-Time Prediction**
   a. Open and run all cells in `modelTesting.ipynb`.
   b. This script will load the trained model from `Models/age_gender_model.keras`.
   c. It will then open your webcam, detect faces, and draw the predicted age and gender on the live video feed.
   d. Press the **'q'** key on the OpenCV window to stop the webcam feed and end the script.

## Model Architecture

The model is a Convolutional Neural Network (CNN) built using the Keras Functional API. It features a shared convolutional base for feature extraction and two separate output heads for the different prediction tasks.

- **Input:** 128x128x1 (Grayscale Image)
- **Convolutional Base:**
  - `Conv2D` (32 filters) -> `MaxPooling2D`
  - `Conv2D` (64 filters) -> `MaxPooling2D`
  - `Conv2D` (128 filters) -> `MaxPooling2D`
  - `Conv2D` (256 filters) -> `MaxPooling2D`
  - `Flatten`
- **Output Heads:**
  - **Gender Head:** `Dense(256)` -> `Dropout(0.3)` -> `Dense(1, activation='sigmoid')`
    - **Loss:** `binary_crossentropy`
  - **Age Head:** `Dense(256)` -> `Dropout(0.3)` -> `Dense(1, activation='relu')`
    - **Loss:** `mae` (Mean Absolute Error)

## Dependencies

This project requires Python 3.11 and the following libraries:

- `python=3.11`
- `numpy`
- `pandas`
- `seaborn`
- `tqdm`
- `matplotlib`
- `tensorflow`
- `keras`
- `opencv-python`
- `jupyter`
