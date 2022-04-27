import cv2
import time
from detector.handDetector import HandDetector

cap = cv2.VideoCapture(0)
start = time.time()
numFrame = 0

hd = HandDetector()

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame,1)
    hd.detect_hands_landmarks(frame)
    if success:
        numFrame += 1
    curr = time.time()
    fps = int(numFrame/(curr-start))
    cv2.putText(frame,str(fps),(20,50), cv2.FONT_HERSHEY_PLAIN, 2, (200,0,200), 2)
    hd.draw_landmarks_on_frame(frame, draw_connections=True)
    
    # count the fingers up
    coord = hd.get_lm_coordinates_for_given_hand(frame)
    fingers_up = []
    tip_lms = [8,12,16,20]
    if len(coord) != 0:
        for tip_lm in tip_lms:
            if coord[tip_lm][2] > coord[tip_lm-2][2]:
                fingers_up.append(0)
            else:
                fingers_up.append(1)
        if coord[4][1] > coord[2][1]:
            fingers_up.append(0)
        else:
            fingers_up.append(1)
    print(sum(fingers_up))
    
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()