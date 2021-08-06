import cv2

cap = cv2.VideoCapture(0)
cascade_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()

    frame = cv2.cvtColor(frame, 0)
    detection = cascade_classifier.detectMultiScale(frame, 1.3, 5)

    if len(detection) > 0:
        (x, y, w, z) = detection[0]
        frame = cv2.rectangle(frame, (x, y), (x + w, y + z), (255, 0, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindow()
