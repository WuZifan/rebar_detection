import cv2

cap = cv2.VideoCapture('rtsp://192.168.8.102:554/onvif/live/1/1')

while True:
    try:
        ret,frame = cap.read()
        cv2.imshow('a',frame)
        cv2.waitKey(1)
    except Exception as e:
        cap = cv2.VideoCapture('rtsp://192.168.8.102:554/onvif/live/1/1')