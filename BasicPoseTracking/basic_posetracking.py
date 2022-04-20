# imports
import cv2
import time
from detectors.poseDetector import pose_detector

# start video streamer
cap = cv2.VideoCapture(0)

# initialize objects
pd = pose_detector()

# initialize variables
num_frame = 0
start_time = time.time()

# loop to read frames
while True:
    success, frame = cap.read()
    if success:
        num_frame += 1
    current_time = time.time()
    fps = int(num_frame / (current_time - start_time))
    cv2.putText(frame, str(fps), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (240,34,223), 2)
    pd.detect_pose(frame)
    # pd.draw_pose_landmarks_on_frame(frame, draw_connections=True)
    # pd.draw_specific_landmarks_on_frame(frame, ids=[8])
    coor_list = pd.get_lm_coordinates(frame, lm_id=8)
    print(coor_list)
    cv2.imshow("image", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# release streamer
cap.release()
cv2.destroyAllWindows()