# 🔬 Skin Cancer Detection using CNNs

> Leveraging Deep Learning for accurate classification of skin lesions.

-----

## 🚀 Overview

This is a deep learning project focused on classifying skin lesions as either **Benign** or **Malignant** using Convolutional Neural Networks (CNNs). This project demonstrates the entire machine learning pipeline, from data loading and preprocessing with augmentation to model building, training, evaluation, and making predictions on new images. It aims to provide a reliable tool for preliminary skin cancer assessment.

-----

## ✨ Features

  * **Data Preprocessing**: Efficiently loads and preprocesses image data using `ImageDataGenerator`, including normalization and augmentation techniques like rotation, shifts, shears, and zooms to enhance model generalization.
  * **Custom CNN Architecture**: Implements a sequential CNN model with multiple convolutional and pooling layers, followed by dense layers and dropout for classification.
  * **Model Training**: Trains the CNN model on a dataset of skin images, utilizing `EarlyStopping` to prevent overfitting and `ReduceLROnPlateau` to optimize the learning rate during training.
  * **Comprehensive Evaluation**: Assesses model performance using standard metrics and visualizations:
      * **Accuracy Plots**: Visualizes training and validation accuracy over epochs.
      * **Loss Plots**: Shows training and validation loss progression.
      * **Classification Report**: Provides precision, recall, and F1-score for each class.
      * **Confusion Matrix**: Clearly displays the true positive, true negative, false positive, and false negative predictions.
  * **Prediction Functionality**: Includes a utility to load a trained model and predict the class (Benign or Malignant) of a new skin lesion image.

-----

## 💻 How to Run Locally

To run this project on your machine, follow these steps:

### 1\. Download the Dataset

The project expects a dataset in a specific structure. You'll need to download an `archive.zip` file containing `train` and `test` directories, each with `Benign` and `Malignant` subfolders.

The provided code uses `!unzip archive.zip -d /content/dataset` which suggests it's designed to run in a Colab-like environment where the zip file is uploaded and unzipped to `/content/dataset`. If running locally, ensure your dataset is unzipped into a folder structure like:

```
your_project_directory/
└── dataset/
    ├── train/
    │   ├── Benign/
    │   └── Malignant/
    └── test/
        ├── Benign/
        └── Malignant/
```

### 2\. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-project-directory> # e.g., cd DermAI
```

*(Replace `<your-repository-url>` with the actual URL of your Git repository and `<your-project-directory>` with your desired project folder name.)*

-----

### 3\. Install Dependencies

Ensure you have Python installed (Python 3.8+ recommended). Then, install the required libraries using pip:

```bash
pip install tensorflow matplotlib seaborn scikit-learn
```

-----

### 4\. Execute the Script

Run the main Python script from your terminal:

```bash
python dipproject.py # Or whatever your main script name is
```

This script will:

  * Unzip the `archive.zip` (if it's in the same directory and you're running in an environment that supports `!unzip` or you've manually unzipped it).
  * Load and preprocess the image data.
  * Display sample images from the training set.
  * Build and compile the CNN model.
  * Train the model.
  * Save the trained model as `skin_cancer.h5`.
  * Display plots for training/validation accuracy and loss, and the confusion matrix.
  * Print a classification report.
  * Finally, it will make a prediction on a sample test image (`/content/dataset/test/Malignant/6300.jpg`) and display the image with its prediction.

-----

## 📸 Screenshots

Here are some of the visualizations and outputs you can expect when running the script:
![image](https://github.com/user-attachments/assets/dd9126c0-df0b-4d53-9e6f-60d9b1096afc)
![image](https://github.com/user-attachments/assets/52aed874-0174-41a2-8c7b-d59fb7af1e76)
![image](https://github.com/user-attachments/assets/a1fb32ea-8a53-4de3-8a18-5325f9c584e7)
![image](https://github.com/user-attachments/assets/e4207d04-2a61-4be8-afc3-282618d82ad9)

-----

## 🏗 Project Structure

```
your_project_directory/
├── dipproject.py               # Main script containing all code
├── archive.zip                 # Zipped dataset (expected by the script)
├── skin_cancer.h5              # Trained model file (generated after first run)
├── README.md                   # This file
└── (optional: images/)         # Directory to store generated plot screenshots
    ├── sample_images.png
    ├── accuracy_plot.png
    ├── loss_plot.png
    ├── confusion_matrix_plot.png
    └── predicted_image.png
```

-----

## 🛠 Tech Stack

  * **Python**: The core programming language.
  * **TensorFlow/Keras**: For building, training, and evaluating the deep learning models.
  * **NumPy**: For numerical operations.
  * **Matplotlib**: For creating static, interactive, and animated visualizations.
  * **Seaborn**: For statistical data visualization, used for the confusion matrix heatmap.
  * **Scikit-learn**: For classification report and confusion matrix generation.

-----

## 🤝 Contributing

Contributions are welcome\! If you have suggestions for improvements, new features, or bug fixes, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and commit them (`git commit -m 'Add new feature'`).
4.  Push to the branch (`git push origin feature/your-feature-name`).
5.  Open a Pull Request.

-----
