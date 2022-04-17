import cv2
import mediapipe as mp
from numpy import array

class HandLandmarkException(Exception):
    """Exception for class hand_detector

    Args:
        Exception (exception): default exception
    """

class hand_Detector:
    
    def __init__(self, 
                 static_image_mode=False, 
                 max_num_hands=2, 
                 min_detection_confidence=0.5, 
                 min_tracking_confidence=0.5) -> None:
        """Constructor to the hand detector class.

        Args:
            static_image_mode (bool, optional): True if detect and track. Defaults to False.
            max_num_hands (int, optional): max number of hands in a single frame. Defaults to 2.
            min_detection_confidence (float, optional): threshold for detection. Defaults to 0.5.
            min_tracking_confidence (float, optional): threshold for tracking. Defaults to 0.5.
        """
        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(static_image_mode=static_image_mode,
                                   max_num_hands=max_num_hands,
                                   min_detection_confidence=min_detection_confidence,
                                   min_tracking_confidence=min_tracking_confidence)
        
    def draw_landmarks_on_frame(self, frame, draw_connections=False) -> array:
        """draw the hand landmarks on frame

        Args:
            frame: the frame to be drawn on
            draw_connections (bool, optional): if True draw connections between landmarks

        Returns:
            Array: return the preprocessed image
        """
        cvtFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(cvtFrame)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                if draw_connections:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
                else:
                    self.mpDraw.draw_landmarks(frame, handLms)
        return frame
    
    def draw_specific_landmarks_on_frame(self, frame, ids) -> None:
        """draw circle on a give id(s)

        Args:
            frame : the frame to be drawn on
            ids (list): list of landmarks to be shown in the image.

        Raises:
            HandLandmarkException: if list is empty.
            HandLandmarkException: if one of the given id(s) doesn't belong to the list of lms.
        """
        if len(ids) > 0 and not all(item in range(1,21) for item in ids) :
            raise HandLandmarkException("The given landmarks are not in the recognized landmark list")
        if len(ids) == 0:
            raise HandLandmarkException("The list of landmarks is empty")
        h, w, _ = frame.shape
        cvtFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(cvtFrame)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for idy, lm_coor in enumerate(handLms.landmark):
                    cx, cy = int(lm_coor.x * w), int(lm_coor.y * h)
                    if idy in ids:
                        cv2.circle(frame, (cx,cy), 5, (255,255,0), cv2.FILLED)
                        
    def get_lm_coordinates_for_given_hand(self, frame, lm_id, handnum=0) -> tuple:
        h, w, _ = frame.shape
        cvtFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(cvtFrame)
        if results.multi_hand_landmarks:
            myHand = results.multi_hand_landmarks[handnum]
            for idy, lm_coor in enumerate(myHand.landmark):
                if idy == lm_id:
                    return (int(lm_coor.x * w), int(lm_coor.y * h))
            