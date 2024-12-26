import cv2
import numpy as np
import base64


def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')


def capture_frame(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise ValueError("Error: Unable to open the RTSP stream.")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Error: Unable to read frame from the RTSP stream.")
    return frame


def base64_to_image(base64_str):
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
