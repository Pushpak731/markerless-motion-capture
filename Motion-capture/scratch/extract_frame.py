import cv2
cap = cv2.VideoCapture('data/video1_source.mp4')
ret, frame = cap.read()
if ret:
    cv2.imwrite('data/video1_frame1.png', frame)
    print("Saved frame 1")
else:
    print("Could not read frame")
cap.release()
