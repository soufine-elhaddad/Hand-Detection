import pickle
import csv
import cv2
import mediapipe as mp
import numpy as np

def load_model(file_path='./model1.p'):
    model_dict = pickle.load(open(file_path, 'rb'))
    model = model_dict['model']
    return model

def initialize_hands_detector():
    return mp.solutions.hands.Hands(static_image_mode=True,max_num_hands=1, min_detection_confidence=0.3)

def draw_hand_landmarks(frame, hand_landmarks, W, H):
    drawing_spec = mp.solutions.drawing_styles.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
    mp.solutions.drawing_utils.draw_landmarks(
        frame,
        hand_landmarks,
        mp.solutions.hands.HAND_CONNECTIONS,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=drawing_spec
    )


# def draw_hand_landmarks(frame, hand_landmarks, W, H):
#     mp.solutions.drawing_utils.draw_landmarks(
#         frame,
#         hand_landmarks,
#         mp.solutions.hands.HAND_CONNECTIONS,
#         mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
#         mp.solutions.drawing_styles.get_default_hand_connections_style()
#     )

def get_normalized_data(hand_landmarks):
    data_list = []
    tempX = []
    tempY = []

    for i in range(len(hand_landmarks.landmark)):
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y

        tempX.append(x)
        tempY.append(y)

    for i in range(len(hand_landmarks.landmark)):
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y
        data_list.append(x - min(tempX))
        data_list.append(y - min(tempY))

    return data_list

def main():
    model = load_model()
    hands = initialize_hands_detector()

    cap = cv2.VideoCapture(0)

    time = 0
    predicted_char = -1
    L_ = []

    while True:
        ret, frame = cap.read()

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                draw_hand_landmarks(frame, hand_landmarks, W, H)

            hand_landmarks = results.multi_hand_landmarks[0]
            data_list = get_normalized_data(hand_landmarks)

            x1 = int(min(hand_landmarks.landmark, key=lambda x: x.x).x * W) - 10
            y1 = int(min(hand_landmarks.landmark, key=lambda x: x.y).y * H) - 10

            x2 = int(max(hand_landmarks.landmark, key=lambda x: x.x).x * W) + 10
            y2 = int(max(hand_landmarks.landmark, key=lambda x: x.y).y * H) + 10

            prediction = model.predict([np.asarray(data_list)])
            predicted_character = prediction

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, str(''.join(predicted_character)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

            if predicted_character == predicted_char:
                time += 1
                if time == 45:
                    L_.append(str(''.join(predicted_character)))
                    L_.append(' ')
                    time = 0
            else:
                time = 0
                predicted_char = predicted_character

            cv2.putText(frame, ''.join(L_), (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        k = cv2.waitKey(1)
        if k == 27:
            break
    
    csv_file_name = "OutputFile.csv"
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(L_)
    cap.release()
    cv2.destroyAllWindows()
main()