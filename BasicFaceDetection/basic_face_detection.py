import cv2
import time
from detector.faceDetector import FaceDetector

def main():
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    fd = FaceDetector()
    num_frame = 0
    while True:
        success, frame = cap.read()
        frame = cv2.flip(frame,1)
        if success:
            num_frame += 1
        current_time = time.time()
        fps = int(num_frame/(current_time-start_time))
        fd.detect_face(frame)
        # fd.draw_bounding_boxes_and_scores(frame)
        fd.draw_bounding_boxes_and_keypoints(frame)
        cv2.putText(frame, str(fps), (20,40), cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,0),2)
        cv2.imshow("Image", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
