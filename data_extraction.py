import os
import cv2
import mediapipe as mp
import csv
import numpy as np
import random

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

data_dir = 'dataset'
train_dir = 'gesture/train/'
test_dir = 'gesture/test/'





def extract_info_to_csv_file(file,dest_dir,label):
    dest_file = os.path.join(dest_dir, label + '.csv')
    img = cv2.imread(file)

    if img is None:
        print("image not found")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img)

    if results.multi_hand_landmarks:
        with open(dest_file,'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            for hand_landmarks in results.multi_hand_landmarks:
                data = []
                for landmark in hand_landmarks.landmark:
                    data.append(landmark.x)
                    data.append(landmark.y)
                    data.append(landmark.z)
                csvwriter.writerow(data)
        print("Extracted "+ label + " successfully!")

def extract_info_from_files(files, src_dir, dest_dir, label):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    for file in files:
        file = os.path.join(src_dir,file)
        extract_info_to_csv_file(file,dest_dir,label)
        
    

def split(base,train_dir,test_dir,test_percent):
    subdirs = [x[0] for x in os.walk(base)]
    subdirs = subdirs[1:]
    for subdir in subdirs:
        files = os.listdir(subdir)
        random.shuffle(files)
        train_index = int(len(files)*(1-test_percent))
        test_index = int(len(files) * test_percent) + train_index

        train_files = files[:train_index]
        test_files = files[train_index:test_index]

        label = os.path.basename(subdir)[3:]
        extract_info_from_files(train_files,subdir,train_dir,label)
        extract_info_from_files(test_files,subdir,test_dir,label)



def split_and_extract_data(data_dir,train_dir,test_dir,test_percent):
    subdirs = [x[0] for x in os.walk(data_dir)]
    subdirs = subdirs[1:]
    for subdir in subdirs:
        split(subdir,train_dir,test_dir,test_percent)


split_and_extract_data(data_dir,train_dir,test_dir,0.3)