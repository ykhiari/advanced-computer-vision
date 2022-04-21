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




















































# import cv2
# import time
# import mediapipe as mp

# cap = cv2.VideoCapture(0)

# start_time = time.time()
# num_frame = 0

# mp_face_detection = mp.solutions.face_detection
# mp_draw_utils = mp.solutions.drawing_utils
# face = mp_face_detection.FaceDetection()

# while True:
#     success, frame = cap.read()
#     if success:
#         num_frame += 1
#     current_time = time.time()
#     frame = cv2.flip(frame,1)
#     h, w, c = frame.shape
#     fps = int(num_frame / (current_time-start_time))
#     cv2.putText(frame, str(fps), (50,70), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
#     cvtFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face.process(cvtFrame)
#     if results.detections:
#         for detection in results.detections:
#             # mp_draw_utils.draw_detection(frame, detection)
#             bbox_normalized = detection.location_data.relative_bounding_box
#             bbox = int(bbox_normalized.xmin * w), int(bbox_normalized.ymin * h),\
#                 int(bbox_normalized.width * w), int(bbox_normalized.height * h)
#             cv2.rectangle(frame, bbox, (222,244,12), 2)
#             cv2.putText(frame, f"{str(round(detection.score[0]*100))}%", (bbox[0],bbox[1]-10), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
#     cv2.imshow("image", frame)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()