
# Hand Gesture Recognition with MediaPipe and Machine Learning

This repository contains code and resources for a machine learning model that is able to recognize and classify hand gestures using MediaPipe, a framework for building and deploying cross-platform multimodal machine learning pipelines.
The model has been trained on a dataset of images of various hand gestures, and is able to predict the gesture being made in a new image. 

## Data

The data for this project was obtained from the [Gesture Recognition Dataset](https://www.kaggle.com/gti-upm/leapgestrecog) on Kaggle. The dataset consists of 10 different hand gestures, each represented by a set of images taken from different angles. The dataset contains a total of 20000 images, with 2000 images for each gesture.

The data was split into a training set and a test set, with 70% of the data being used for training and 30% being used for testing. 

## Data Extraction

The `data_extraction.py` file is used to split the dataset into training and testing sets and extract hand landmark information using MediaPipe.

The data is first split into a training set and a test set, with a specified percentage of the data being used for training and the remainder being used for testing. The training and test sets are saved to separate directories.

Next, the hand landmark information is extracted from each image using MediaPipe. The hand landmark information consists of the x and y coordinates of the landmarks on the hand, as well as the visibility of each landmark. This information is saved to a CSV file for each image.

The CSV files can then be used as input to the machine learning model for training and evaluation.

## Training

The `train.py` file is used to train the machine learning model on the training data and evaluate its performance on the testing data.

To train the model, the `train.py` file loads the training data and uses it to update the model's parameters. The model's performance is then evaluated on the testing data to assess its generalization to unseen data.

The `train.py` file also supports various training options, such as the ability to specify the number of epochs to train for and the learning rate.

After training is complete, the `train.py` file saves the trained model to a file for future use.

## Testing

The `test.py` file is used to test the trained machine learning model on real-time hand gestures captured through the webcam using OpenCV.

To use the `test.py` file, the trained model must first be saved to a file. The `test.py` file then loads the trained model and uses it to make predictions on hand gestures captured through the webcam in real-time.

The `test.py` file displays the label name for the predicted hand gesture on the frame of the video, allowing the user to see the model's prediction in real-time.

## Dependencies

The following dependencies are required to run the code in this repository:

- Python 3.x
- NumPy
- Pandas
- scikit-learn
- TensorFlow or PyTorch
- MediaPipe
- OpenCV (for the `test.py` file)

## Usage

To extract hand landmark information from the images and save it to CSV files, run the following command:

```bash
python data_extraction.py
```

To train the model on the training data and evaluate its performance on the test data, run the following command:

```bash
python train.py
```

To test the trained model on real-time hand gestures captured through the webcam, run the following command:


```bash
python test.py

```

We hope you find this project useful! If you have any questions or suggestions, please don't hesitate to reach out.




