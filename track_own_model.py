import cv2
import numpy as np
from ultralytics import YOLO

def air_canvas_with_own_model():
    model = YOLO("./best.pt", task="pose")
    cap = cv2.VideoCapture(0)
    
    default_color = (0, 0, 255)
    drawing_color = (255, 0, 0)
    
    canvas = None
    x1, y1 = 0, 0
    
    while True:
        ret, original_frame = cap.read()
        
        if not ret:
            break
            
        if canvas is None:
            canvas = np.zeros_like(original_frame)
            
        original_height, original_width = original_frame.shape[:2]
        input_size = (224, 224)
        resized_frame = cv2.resize(original_frame, input_size) 
        results = model(resized_frame)
        if results and len(results) > 0:
            keypoints = results[0].keypoints.data[0] if results[0].keypoints is not None else None 
            if keypoints is not None:
                scale_x = original_width / input_size[0]
                scale_y = original_height / input_size[1]
                index_finger_keypoint_idx = 8       
                if len(keypoints) > index_finger_keypoint_idx:
                    index_point = keypoints[index_finger_keypoint_idx]
                    confidence = float(index_point[2].item()) 
                    if confidence >= 0.95:
                        x2 = int(index_point[0] * scale_x)
                        y2 = int(index_point[1] * scale_y)
                        cv2.circle(original_frame, (x2, y2), 15, drawing_color, -1)
                        if x1 != 0 and y1 != 0:
                            cv2.line(canvas, (x1, y1), (x2, y2), drawing_color, 5)
                        x1, y1 = x2, y2
                    else:
                        x1, y1 = 0, 0
                for point in keypoints:
                    original_x = int(point[0] * scale_x)
                    original_y = int(point[1] * scale_y)
                    if float(point[2].item()) >= 0.95:
                        cv2.circle(original_frame, (original_x, original_y), 5, default_color, -1)
        result = cv2.addWeighted(original_frame, 0.5, canvas, 0.5, 0)
        cv2.imshow("Hand Keypoints Drawing", result)  
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            canvas = np.zeros_like(original_frame)
            print("Canvas cleared")
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    air_canvas_with_own_model()
