# Real-Time Emotion Recognition Using Webcam

## Overview

This project enables real-time recognition of human emotions through your webcam using deep learning. The system is trained on the FER2013 dataset and can classify faces into seven emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise. The solution includes a Jupyter notebook for training and evaluation, and a Python script for real-time emotion recognition using your webcam.

## Features

- **Emotion Detection:**  
  The model classifies faces as one of seven emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, or Surprise.
- **Transfer Learning:**  
  EfficientNet-B2 pretrained on ImageNet is fine-tuned for emotion recognition.
- **Visual Feedback:**  
  Training progress, accuracy, and loss are logged and visualized with TensorBoard. Confusion matrix is generated for test results.
- **Real-Time Inference:**  
  Use your webcam to detect emotions live with the included `webcam.py` script.
- **Metrics & Logging:**  
  All results, including class names, confusion matrix, and training logs, are saved for review.

## How It Works

1. **Dataset Download:** The FER2013 dataset is automatically downloaded from Kaggle using KaggleHub when you run the notebook.
2. **Preprocessing & Data Loading:** Images are resized, normalized, and split into training and test sets.
3. **Model Training:** EfficientNet-B2 is fine-tuned to classify emotions. Weighted loss is used to address class imbalance.
4. **Evaluation:** The model is evaluated on the test set, and a confusion matrix is generated.
5. **Real-Time Prediction:** Use the webcam script to predict emotions from live video.

## Dataset

No manual download required! The dataset is fetched automatically by the notebook.

- **Source:** [Kaggle - FER2013](https://www.kaggle.com/datasets/msambare/fer2013)

## Technologies Used

- **PyTorch** – For building and training the model.
- **TorchVision** – For image transforms and data loading.
- **TensorBoard** – To visualize the training process.
- **OpenCV** – For real-time face detection and webcam integration.
- **Other libraries:** Pillow, tqdm, mlxtend, torchmetrics, kagglehub.

## Local Setup and Usage

**You don’t need to download the dataset manually! The notebook will handle it for you.**

### 1. Clone the repository and install dependencies

This project includes a `requirements.txt` file listing all necessary Python packages. Install them with:

```bash
git clone https://github.com/hrishikeshChandi/emotions-recognition.git
cd emotions-recognition-main
pip install -r requirements.txt
```

### 2. Run the notebook for training and evaluation

Open `main.ipynb` in Jupyter Notebook or VS Code and run all cells. The dataset will be downloaded automatically, the model will be trained, and results will be saved in the `emotions_results (b2_5e-4)/` folder.

### 3. Run the webcam script for real-time emotion recognition

After training, launch the webcam app:

```bash
python webcam.py
```

Look at your webcam and see your detected emotion in real time.

## Project Structure

- `main.ipynb` – Notebook for training and evaluation (run locally).
- `webcam.py` – Real-time emotion recognition using your webcam.
- `requirements.txt` – List of all required Python packages.
- `emotions_results (b2_5e-4)/` – Stores trained model, logs, and evaluation metrics.
- `haar_face.xml` – Haar cascade for face detection.

## Results

- **Test Set Performance (FER2013, Best Model at Epoch 9):**
  - **Loss:** 0.91838
  - **Accuracy:** 70.03%
- **Confusion Matrix:**
  - Saved as `emotions_results (b2_5e-4)/confusion_matrix.png` after evaluation.
- **Model Checkpoints:**
  - Best model (epoch 9) and CPU-compatible model are saved for deployment.

> **Note:**
> This project is still under active improvement. The current loss and accuracy reflect the best model so far, but further enhancements to model architecture, data augmentation, and training strategy are planned. Expect improved results in future updates!

These results were obtained on the FER2013 test set after training the model using the steps above. Actual performance may vary depending on hardware and training duration.

## Notes

- **No need to manually download the dataset**—the notebook will do it for you.
- The pipeline works best with a local GPU, but will also run on CPU (slower).
- For real-time emotion recognition, ensure your webcam is connected and accessible.

## License

This project is licensed under the MIT License.
