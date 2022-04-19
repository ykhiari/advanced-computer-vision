import cv2
import mediapipe as mp

class PoseLandmarkException(Exception):
    """Exception for class pose_detector

    Args:
        Exception (exception): default exception
    """
    
class pose_detector:
    
    def __init__(self,
                static_image_mode=True,
                model_complexity=2,
                smooth_landmarks=True,
                enable_segmentation=True,
                smooth_segmentation=True,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6) -> None:
        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(
                                    static_image_mode=static_image_mode,
                                    model_complexity=model_complexity,
                                    smooth_landmarks=smooth_landmarks,
                                    enable_segmentation=enable_segmentation,
                                    smooth_segmentation=smooth_segmentation,
                                    min_detection_confidence=min_detection_confidence,
                                    min_tracking_confidence=min_tracking_confidence)
        
    def detect_pose(self, frame):
        cvtFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(cvtFrame)
        
    def draw_pose_landmarks_on_frame(self, frame, draw_connections=False):
        if self.results.pose_landmarks:
            if draw_connections:
                self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
            else:
                self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks)
                
    def draw_specific_landmarks_on_frame(self, frame, ids) -> None:
        if len(ids) > 0 and not all(item in range(1,33) for item in ids) :
            raise PoseLandmarkException("The given landmarks are not in the recognized landmark list")
        if len(ids) == 0:
            raise PoseLandmarkException("The list of landmarks is empty")
        h, w, _ = frame.shape
        if self.results.pose_landmarks:
            for idy, lm_coor in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm_coor.x * w), int(lm_coor.y * h)
                if idy in ids:
                    cv2.circle(frame, (cx,cy), 5, (255,255,0), cv2.FILLED)
                        
    def get_lm_coordinates(self, frame, lm_id) -> tuple:
        h, w, _ = frame.shape
        if self.results.pose_landmarks:
            for idy, lm_coor in enumerate(self.results.pose_landmarks.landmark):
                if idy == lm_id:
                    return (int(lm_coor.x * w), int(lm_coor.y * h))