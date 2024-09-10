
# Fire Detection Using Deep Learning

## Project Overview

This project focuses on building a robust fire detection model using convolutional neural networks (CNNs), MobileNetV2, and other architectures to accurately identify fire, smoke, and non-fire images. The goal is to improve early fire detection to aid in rapid response and mitigation.

## Dataset

We use a publicly available dataset of forest fire, smoke, and non-fire images, sourced from Kaggle:
- **Dataset**: [Forest Fire, Smoke, and Non-Fire Image Dataset](https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset/data)

The dataset is split into three categories: fire, smoke, and non-fire images. These images are used to train the models to classify scenes into one of these three categories.

## Data Loading

We utilize Kaggle's API to download the dataset into the project directory. Ensure that you have installed the Kaggle API and set it up with the appropriate credentials.

### Example of loading the dataset in the code:
```bash
!pip install kaggle
!kaggle datasets download -d amerzishminha/forest-fire-smoke-and-non-fire-image-dataset
!unzip forest-fire-smoke-and-non-fire-image-dataset.zip
```

## Models Used

The following architectures are employed for fire detection:

1. **Convolutional Neural Network (CNN)**:
   - A custom-built CNN with several layers of convolution, max pooling, dropout, and dense layers.

2. **MobileNetV2**:
   - A pre-trained MobileNetV2 model with fine-tuning. This model leverages transfer learning to boost performance with limited computational resources.

3. **Artificial Neural Network (ANN)**:
   - A fully connected ANN model trained on the flattened image data.

4. **Recurrent Neural Network (RNN)**:
   - A combination of Conv2D and LSTM/GRU layers for capturing both spatial and sequential patterns in the data.

## Hyperparameter Tuning

Keras Tuner is used to optimize the hyperparameters of the CNN, ANN, MobileNetV2, and RNN models. The hyperparameter search is based on the `Hyperband` search strategy, with a maximum of 4 trials per model to identify the optimal learning rate and other parameters.

```python
# Example: Keras Tuner for CNN
tuner = kt.Hyperband(
    cnn_model,
    objective='val_accuracy',
    max_trials=4,
    max_epochs=10,
    factor=3,
    directory='kt_search',
    project_name='cnn_tuning'
)
tuner.search(train_generator, validation_data=validation_generator, epochs=10)
```

## Performance Evaluation

Each model is evaluated using the following metrics:
- **Accuracy**: Measures the overall correctness of the model.
- **Confusion Matrix**: Provides detailed insights into the types of misclassifications.
- **Classification Report**: Shows precision, recall, F1-score, and support for each class (fire, smoke, non-fire).

Example of training and evaluating a model:
```python
train_and_evaluate(best_cnn_model, "Best CNN Model")
```

## Results

The results from each model, including accuracy, confusion matrix, and classification report, are saved and displayed during training. The best-performing model is saved in the `output` directory.

## Installation

### Prerequisites

To run this project, you'll need the following libraries:

- `tensorflow`
- `keras`
- `opencv-python`
- `seaborn`
- `matplotlib`
- `keras-tuner`
- `sklearn`
- `kaggle`

Install the required packages with:

```bash
pip install tensorflow keras opencv-python seaborn matplotlib keras-tuner scikit-learn kaggle
```

## How to Run the Project

1. **Download the Dataset**: Download the dataset from Kaggle using the Kaggle API or manually from [this link](https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset/data).
   
2. **Prepare the Data**: Load the dataset into the directory using the provided loading instructions.

3. **Train the Models**: Run the training script to train and evaluate the models.

```bash
python train_models.py
```

4. **Evaluate Results**: Check the results in the `output` folder for saved models and evaluation metrics.

## Future Work

1. **Additional Data Augmentation**: Explore more advanced augmentation techniques to improve model robustness.
2. **Additional Models**: Explore other architectures like EfficientNet, ResNet, or even Transformers.
3. **Deploying the Model**: Integrate the model into a real-time fire detection system using edge computing devices such as Raspberry Pi.

## Conclusion

This project demonstrates the application of deep learning for fire and smoke detection in real-world forest environments. The integration of multiple models and hyperparameter tuning improves detection accuracy and robustness, making this project a valuable tool for early fire detection.

---

This README covers all essential aspects of the project, from data loading, model building, tuning, and evaluation to future work. It aligns with your previous project structure, ensuring consistency across your repositories.