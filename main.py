import argparse
import requests
from deepface import DeepFace
from helpers import *
import numpy as np
from PIL import Image

# Default values
DEFAULT_RTSP_URL = 'rtsp://testtest:000000@192.168.1.67:554/stream1'
DEFAULT_SITE = 'tss'

# Argument parsing
parser = argparse.ArgumentParser(description='Process RTSP stream and perform face verification.')
parser.add_argument('--cam.ip', dest='rtsp_url', default=DEFAULT_RTSP_URL,
                    help=f'RTSP URL of the camera (default: {DEFAULT_RTSP_URL})')
parser.add_argument('--cam.site', dest='site', default=DEFAULT_SITE,
                    help=f'Site identifier (default: {DEFAULT_SITE})')
args = parser.parse_args()

# Assign variables from parsed arguments
RTSP_URL = args.rtsp_url
SITE = args.site

URL = "http://192.168.1.9:3362/v1/api/employees/imgs"
POSTURL = "http://192.168.1.9:3362/v1/api/attendance/notifactions"
IMAGES = requests.get(URL).json()

while True:
    frame = capture_frame(RTSP_URL)
    frame64 = frame_to_base64(frame)

    if frame.dtype != np.uint8:
        frame = (frame * 255).astype(np.uint8)

    image = Image.fromarray(frame)
    image.save(f'frame.jpg')

    for i in IMAGES:
        db_image = base64_to_image(i['img'])
        result = DeepFace.verify(
            img1_path=f'frame.jpg',
            img2_path=db_image,
            model_name="VGG-Face",
            detector_backend="opencv",
            distance_metric="cosine",
            enforce_detection=False,
            align=True,
            normalization="base"
        )

        if result['verified'] and i['site'] != SITE:
            print(f"New Violation! Name: {i['employeeName']}, Found In: {SITE}\n")

            payload = {
                'area': SITE,
                'employeeId': i['_id'],
                'img': frame64
            }

            response = requests.post(POSTURL, json=payload, headers={'Content-Type': 'application/json'})
            print(f'Response Status: {response.status_code}')

        if cv2.waitKey(1) & 0xFF == 27:
            print("Escape key pressed. Exiting...")
            break

    # Check for 'Escape' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == 27:
        print("Escape key pressed. Exiting...")
        break
