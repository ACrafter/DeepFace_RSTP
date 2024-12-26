import cv2

USERNAME = 'testtest'
PASSWORD = '000000'

RTSP_URL = f"rtsp://{USERNAME}:{PASSWORD}@192.168.1.67:554/stream2"

cap = cv2.VideoCapture(RTSP_URL)
if not cap.isOpened():
    raise ValueError("Error: Unable to open the RTSP stream.")

cv2.namedWindow('live cam', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    cv2.imshow('GreenMEA', frame)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
