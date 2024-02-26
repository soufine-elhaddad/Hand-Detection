import os
import pickle
import mediapipe as mp
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# import numpy as np

mp_hands = mp.solutions.hands

DATA_DIR = './data'
OUTPUT_FILE = 'data.pickle'

def process_image(img_path, hands_detector):
    data_list = []

    tempX = []
    tempY = []

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands_detector.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y

                tempX.append(x)
                tempY.append(y)

            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                data_list.append(x - min(tempX))
                data_list.append(y - min(tempY))

    return data_list

def main():
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    data = []
    labels = []

    for dir_ in os.listdir(DATA_DIR):
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            img_full_path = os.path.join(DATA_DIR, dir_, img_path)
            data_list = process_image(img_full_path, hands)

            if data_list:
                data.append(data_list)
                labels.append(dir_)

    hands.close()
    
    def train_random_forest_classifier(tempXtrain, y_train):
        model = RandomForestClassifier()
        model.fit(tempXtrain, y_train)
        return model
    
    def evaluate_model(model, tempXtest, y_test):
        y_predict = model.predict(tempXtest)
        score = accuracy_score(y_predict, y_test)
        return score
    
    
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    trained_model = train_random_forest_classifier(x_train, y_train)

    accuracy = evaluate_model(trained_model, x_test, y_test)

    print('{}% of samples were classified correctly!'.format(accuracy * 100))


    with open('model1.p', 'wb') as f:
        pickle.dump({'model': trained_model}, f)


if __name__ == "__main__":
    main()

