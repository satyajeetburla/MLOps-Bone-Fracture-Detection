# MLOps-Bone-Fracture-Detection


## Overview
This project aims to develop an end-to-end machine learning pipeline for classifying bone fractures from X-ray images. Leveraging MLOps practices with DVC, the solution was deployed on both Azure and AWS. The primary objective was to build a comprehensive MLOps project, using fracture classification as a simple yet effective demonstration to implement the entire pipeline.


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
    
3. **Activate the Environment:**

   Switch to the newly created fracture environment:
   
    ```bash
   conda activate fracture
    ```
    
4. **Install the Dependencies:**

   Install all required dependencies using the requirements.txt file:
   
   ```bash
   pip install -r requirements.txt
   ```
    
5. **Launch the Application:**

   Execute the application with:
   
   ```bash
   python app.py
   ```
   "app.py" is a Flask application that facilitates training and prediction for a machine learning model, specifically for       image classification tasks.

      
6. **Access the Web Interface:**

   Open your web browser and go to the local host address and port displayed to use the application.
   
   ```bash
   For me it was : http://127.0.0.1:5000
   ```

![Flask Interface](https://github.com/satyajeetburla/MLOps-Bone-Fracture-Detection/blob/main/img/flask%20interface.png)



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

### Some Other Scripts

- **template.py:**  
  This script sets up a standardized folder structure essential for developing the project. It helps maintain consistency across the workflow, making the project easier to manage, scale, and expand. 

- **utils/common.py:**  
    Contains reusable utility functions for tasks like configuration management, file manipulation, logging, and data encoding. The functions are decorated with `@ensure_annotations`, ensuring that they adhere to specified type annotations for robust development and deployment. This makes debugging easier in large machine learning projects and ensures code consistency.

# AWS and Azure CI/CD Deployment with GitHub Actions

## AWS Deployment Steps

### 1. Log in to the AWS Console

### 2. Create an IAM User for Deployment

   Provide the new IAM user with appropriate access permissions:
   
   - **EC2 Access:** Required for launching and managing virtual machines.
   - **ECR (Elastic Container Registry):** Enables you to save Docker images within AWS.

   **Deployment Process:**
   - Build the Docker image from the source code.
   - Push the Docker image to ECR.
   - Launch an EC2 instance.
   - Pull the Docker image from ECR onto the EC2 instance.
   - Start your Docker image on the EC2 instance.

   **Policy Recommendations:**
   - `AmazonEC2ContainerRegistryFullAccess`
   - `AmazonEC2FullAccess`

### 3. Create an ECR Repository

   Create a repository to store and manage Docker images. Keep the URI handy for use during deployment:
   - Example: `566373416292.dkr.ecr.us-east-1.amazonaws.com/chicken`

### 4. Launch an EC2 Machine (Ubuntu)

### 5. Install Docker on the EC2 Machine

   **Optional Updates:**
   ```bash
   sudo apt-get update -y
   sudo apt-get upgrade
   ```
   **Docker Installation Commands:**
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker ubuntu
   newgrp docker
   ```

### 6. Configure EC2 as a Self-Hosted Runner
   Navigate to Settings > Actions > Runners > New Self-Hosted Runner, select your OS, and follow the step-by-step       instructions.

### 7. Set Up GitHub Secrets
   Configure the following secrets in GitHub Actions:

   AWS_ACCESS_KEY_ID
   AWS_SECRET_ACCESS_KEY
   AWS_REGION: e.g., us-east-1
   AWS_ECR_LOGIN_URI: e.g., 566373416292.dkr.ecr.ap-south-1.amazonaws.com
   ECR_REPOSITORY_NAME: e.g., simple-app
   
# Azure Deployment with GitHub Actions

## Save Credentials
Make sure your Azure credentials are securely stored to allow access to the container registry.

## Run These Commands

### Build and Push Docker Image

1. **Build the Docker Image:**
   ```bash
   docker build -t chickenapp.azurecr.io/chicken:latest .
   ```    
2. **Log in to Azure Container Registry:**
   ```bash
   docker login chickenapp.azurecr.io
   ```
3. **Push the Docker Image:**
   ```bash
   docker push chickenapp.azurecr.io/chicken:latest
   ```
Deployment Process
1. **Build the Docker Image:**
   Create a Docker image from the project's source code.

2. **Push to Registry:**
   Upload the Docker image to the Azure container registry.

3. **Launch Web App Server:**
   Create or launch a Web App server on Azure.

4. **Pull and Run Docker Image:**
   Retrieve the Docker image from the registry and run it on the Azure Web App server.

