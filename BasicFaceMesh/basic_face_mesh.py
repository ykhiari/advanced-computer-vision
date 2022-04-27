import cv2
import time
from cv2 import cvtColor
import mediapipe as mp

mpFMesh = mp.solutions.face_mesh
mpDraw = mp.solutions.drawing_utils
face_mesh = mpFMesh.FaceMesh()
drawing_spec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
mp_drawing_styles = mp.solutions.drawing_styles

def main():
    cap = cv2.VideoCapture(0)
    start = time.time()
    num_frame = 0
    while True:
        success, frame = cap.read()
        if success:
            num_frame += 1
        curr = time.time()
        fps = int(num_frame/(curr-start))
        cv2.flip(frame, 1)
        cvtFrame = cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(cvtFrame)
        if results.multi_face_landmarks:
            for face_lms in results.multi_face_landmarks:
                mpDraw.draw_landmarks(image=frame,
                                        landmark_list=face_lms,
                                        # connections=mpFMesh.FACEMESH_TESSELATION,
                                        landmark_drawing_spec=drawing_spec)
        cv2.putText(frame, str(fps), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.imshow("image", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()