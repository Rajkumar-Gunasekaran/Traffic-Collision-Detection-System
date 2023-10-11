import cv2
import numpy as np

# Load the pre-trained car detection model
car_cascade = cv2.CascadeClassifier('assets/haarcascade_car.xml') 

video_source = "assets/car_-_2165 (Original).mp4"
video_capture = cv2.VideoCapture(video_source)

# Define a function to detect and track cars in each frame
def track_cars(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return frame

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Call the track_cars function to detect and track cars in the frame
    output_frame = track_cars(frame)

    cv2.imshow('Car Tracking', output_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
        break


video_capture.release()
cv2.destroyAllWindows()
