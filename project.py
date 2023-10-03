import cv2
def calculate_centroid(contour):
    M = cv2.moments(contour)
    area = M["m00"]
    if area > 0:
        cx = int(M["m10"] / area)
        cy = int(M["m01"] / area)
        return cx, cy
    else:
        return None, None

video_capture = cv2.VideoCapture(0)
background_subtractor = cv2.createBackgroundSubtractorMOG2()
tracked_objects = {}
object_id_counter = 0 

while True:
    ret, frame = video_capture.read()

    if not ret:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fg_mask = background_subtractor.apply(gray_frame)

    _, binary_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = calculate_centroid(contour)

        if cx is not None and cy is not None:
            object_id_counter += 1
            object_id = object_id_counter 
            tracked_objects[object_id] = (cx, cy)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    cv2.imshow('Object Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()
