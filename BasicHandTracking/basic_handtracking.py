import cv2
import time
from detectors.handDetector import hand_Detector

cap = cv2.VideoCapture(0)
numer_of_frames = 0
start_time = time.time()
hand_detector = hand_Detector()

while True:
    success, frame = cap.read()
    if success:
        numer_of_frames += 1
    curent_time = time.time()
    fps = int(numer_of_frames / (curent_time - start_time))
    cv2.putText(frame, str(fps), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
    hand_detector.detect_hands_landmarks(frame)
    # frame = hand_detector.draw_landmarks_on_frame(frame, draw_connections=True)
    # hand_detector.draw_specific_landmarks_on_frame(frame, ids=range(1,21))
    ret_list = hand_detector.get_lm_coordinates_for_given_hand(frame, lm_id=4)
    print(ret_list)
    cv2.imshow("image", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()