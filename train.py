import os
import numpy as np
import csv
import random
import tensorflow as tf
import tensorflow.keras as keras
from keras import layers
from sklearn.preprocessing import OneHotEncoder


def cnn_model(input_shape, num_classes):
    model = keras.Sequential([

        layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    print("Started Compiling")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Model Compiled....")
    return model

def extract_data_and_label(dataset):
    data = []
    labels = []
    for p in dataset:
        data.append(p[0])
        labels.append(p[1])
    return data, labels

def extract_info_from_csv(csv_file):
    label = csv_file.split('/')[-1].split('.')[0]
    dataset = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file,quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            dataset.append([row,label])
    return dataset

def extract_data(base):
    dataset = []
    for file in os.listdir(base):
        dataset += extract_info_from_csv(os.path.join(base,file))
    random.shuffle(dataset)
    return dataset





train_dataset_csv = "gesture/train"
test_dataset_csv = "gesture/test"

train_data = extract_data(train_dataset_csv)
test_data = extract_data(test_dataset_csv)

train_data, train_labels = extract_data_and_label(train_data)
test_data, test_labels = extract_data_and_label(test_data)

labels = np.unique(train_labels)
input_shape = len(train_data[0])
num_classes = len(labels)

x_train = np.array(train_data)
x_test = np.array(test_data)

y_train = OneHotEncoder().fit_transform(np.array(train_labels).reshape(-1, 1)).toarray()
y_test = OneHotEncoder().fit_transform(np.array(test_labels).reshape(-1, 1)).toarray()

my_model = cnn_model(input_shape,num_classes)
my_model.fit(x_train, y_train ,epochs=200, batch_size=64, validation_data=(x_test,y_test))
my_model.save("gesture_model.h5")
