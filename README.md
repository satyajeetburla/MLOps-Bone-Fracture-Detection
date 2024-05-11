# MLOps-Bone-Fracture-Detection


## Overview

# How to Run

### Steps:

1. **Clone the Repository:**

   Clone the project repository using the following command:

   ```bash
   git clone https://github.com/entbappy/Chicken-Disease-Classification--Project
   ```
2. **Create a Conda Environment:**

   Inside the project directory, set up a new Conda environment named fracture with Python 3.8:
    ```bash
   conda create -n fracture python=3.8 -y
   ```
3.**Activate the Environment:**

   Switch to the newly created fracture environment:
   
    ```bash
   conda activate fracture
      ```
4.**Install the Dependencies:**
   Install all required dependencies using the requirements.txt file:
    ```bash
   pip install -r requirements.txt
      ```
5.**Launch the Application:**

   Execute the application with:
   
    ```bash
   python app.py
      ```
      
6.**Access the Web Interface:**

   Open your web browser and go to the local host address and port displayed to use the application.
   
    ```bash
   For me it was : 127.0.0.1
       ```
## Configuration Files

- **Config.yaml:** Centralized storage for global project settings and paths.
- **Params.yaml:** Holds the model parameters for easy tuning and management.

## Pipeline Components

1. **Data Ingestion:**  
   `data_ingestion.py` provides the `DataIngestion` class, which handles data acquisition, extraction, and preparation for consistent downstream processing.

2. **Base Model Preparation:**  
   `prepare_base_model.py` defines the `PrepareBaseModel` class to load, extend, and update pre-trained models using configurable input layers and freezing.

3. **Callback Preparation:**  
   `prepare_callbacks.py` contains the `PrepareCallback` class for efficient TensorBoard logging and model checkpointing.

4. **Training Pipeline:**  
   `training.py` uses the `Training` class to encapsulate model training while ensuring efficient data generation and callback management.

5. **Evaluation Pipeline:**  
   `evaluation.py` contains the `Evaluation` class to load, validate, and score trained models.

6. **Prediction Pipeline:**  
   `prediction.py` provides the `PredictionPipeline` class to load the model, preprocess input images, and return predictions.

## MLOps Tools

**DVC (Data Version Control):**  
Utilized for tracking pipeline changes via `dvc.yaml`. Run `dvc init` to initialize DVC, then configure your stages. Use `dvc repro` to rebuild only those stages that have changed.

## How to Use



