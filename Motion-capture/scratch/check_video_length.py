import cv2
cap = cv2.VideoCapture('data/video1_source.mp4')
print(f"Total frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
cap.release()
