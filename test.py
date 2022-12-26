import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

label = ['C','DOWN','FIST_MOVED','FIST','INDEX','L','OK','PALM_MOVED','PALM','THUMB']
model = load_model('gesture_model.h5')


def landmark_to_vector(landmarks):
    vector = []
    for landmark in landmarks:
        vector.append(landmark.x)
        vector.append(landmark.y)
        vector.append(landmark.z)
    return vector


cap = cv2.VideoCapture(0)
while cap.isOpened():
    status, frame = cap.read()
    if status:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        outcomes = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if outcomes.multi_hand_landmarks:
            for hand_landmarks in outcomes.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
                vector = landmark_to_vector(hand_landmarks.landmark)
                vector = np.array(vector).reshape(1, 63)
                prediction = model.predict(vector)
                index = np.argmax(prediction)
                cv2.putText(frame, label[index],(20,40),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) == ord('q'):
            break