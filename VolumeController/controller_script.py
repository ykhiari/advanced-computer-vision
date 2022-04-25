import cv2
import math
import numpy as np
from detectors.handDetector import HandDetector
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap = cv2.VideoCapture(0)
hd = HandDetector()
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume_range = volume.GetVolumeRange()
min_vol = volume_range[0]
max_vol = volume_range[1]
while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    hd.detect_hands_landmarks(frame)
    lm_list = hd.get_lm_coordinates_for_given_hand(frame)
    if len(lm_list) != 0:
        point1 = (lm_list[4][1], lm_list[4][2])
        point2 = (lm_list[8][1], lm_list[8][2])
        cx, cy = (point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2 
        cv2.circle(frame, point1, 8, (255,0,0), cv2.FILLED)
        cv2.circle(frame, point2, 8, (255,0,0), cv2.FILLED)
        cv2.circle(frame, (cx,cy), 8, (255,0,0), cv2.FILLED)
        cv2.line(frame, point1, point2, (255,0,0), 3)
        dist = math.hypot(point1[0]-point2[0],point1[1]-point2[1])
        vol = np.interp(dist, [50,350], [min_vol, max_vol])
        volume.SetMasterVolumeLevel(vol, None)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()