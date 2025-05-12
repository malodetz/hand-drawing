import cv2
import mediapipe as mp
import numpy as np


def air_canvas():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    drawing_color = (255, 0, 0)
    canvas = None
    x1, y1 = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break
        if canvas is None:
            canvas = np.zeros_like(frame)
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                index_finger_tip = hand_landmarks.landmark[8]
                h, w, c = frame.shape
                x2, y2 = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                index_finger_mcp = hand_landmarks.landmark[5]
                if index_finger_tip.y < index_finger_mcp.y:
                    cv2.circle(frame, (x2, y2), 15, drawing_color, -1)
                    if x1 != 0 and y1 != 0:
                        cv2.line(canvas, (x1, y1), (x2, y2), drawing_color, 5)
                    x1, y1 = x2, y2
                else:
                    x1, y1 = 0, 0
        result = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
        cv2.imshow("Air Canvas", result)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            canvas = np.zeros_like(frame)
            print("Canvas cleared")
        elif key == ord("q"):
            print("Exiting...")
            break
    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    air_canvas()
