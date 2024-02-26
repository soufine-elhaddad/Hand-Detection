import os
import cv2

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def collect_data(class_index, data_dir='./data', number_of_classes=10, dataset_size=100):
    class_dir = os.path.join(data_dir, str(class_index))
    create_directory(class_dir)

    cap = cv2.VideoCapture(0)

    print(f'Collecting data for class {class_index}')

    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'Ready? Press "Q" ! : ){class_index}', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        image_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(image_path, frame)

        counter += 1
        k = cv2.waitKey(1)
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    
    

if __name__ == "__main__":
    DATA_DIR = './data'
    create_directory(DATA_DIR)

    NUMBER_OF_CLASSES = 10
    DATASET_SIZE = 100

    for class_index in range(NUMBER_OF_CLASSES):
        collect_data(class_index, DATA_DIR, NUMBER_OF_CLASSES, DATASET_SIZE)
