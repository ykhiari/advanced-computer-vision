import cv2
import mediapipe as mp

class FaceDetector:
    
    def __init__(self, model_selection=1, min_detection_confidence=0.5):
        self.mp_face_detector = mp.solutions.face_detection
        self.face = self.mp_face_detector.FaceDetection(model_selection=model_selection,
                                          min_detection_confidence=min_detection_confidence)
        self.drawFace = mp.solutions.drawing_utils
        
    def detect_face(self, frame):
        cvtFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.face.process(cvtFrame)
        
    def _get_normalized_bounding_boxes(self):
        norm_bbox = []
        if self.results.detections:
            for detection in self.results.detections:
                norm_bbox.append(detection.location_data.relative_bounding_box)
        return norm_bbox
    
    def _get_confidence_scores(self):
        scores = []
        if self.results.detections:
            for detection in self.results.detections:
                scores.append(round(detection.score[0]*100))
        return scores
    
    def _get_bounding_boxes(self, frame):
        h, w, _ = frame.shape
        bbox = []
        norm_bbox = self._get_normalized_bounding_boxes()
        for n_box in norm_bbox:
            bbox.append((int(n_box.xmin * w), int(n_box.ymin * h),\
                int(n_box.width * w), int(n_box.height * h)))
        return bbox
    
    def draw_bounding_boxes_and_scores(self, frame):
        bboxes = self._get_bounding_boxes(frame)
        scores = self._get_confidence_scores()
        for i, box in enumerate(bboxes):
            cv2.rectangle(frame, box, (222,244,12), 2)
            cv2.putText(frame, f"{str(scores[i])}%", (box[0],box[1]-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,255), 2)
            
    def draw_bounding_boxes_and_keypoints(self, frame):
        for detection in self.results.detections:
            self.drawFace.draw_detection(frame, detection)