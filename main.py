import cv2

from matplotlib import pyplot as plt


cap = cv2.VideoCapture(0)

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=0)

while True:
    ret, frame = cap.read()
    height,width, _ = frame.shape

    roi = frame[100: 900, 100: 900]


    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:

        area = cv2.contourArea(cnt)
        if area > 7000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 225, 0), 3)
            cv2.putText(frame, "Status: {}".format('movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)



    cv2.imshow('mask', mask)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

        cap.release()
        cv2.destroyAllWindows