import cv2

cap = cv2.VideoCapture('test.mp4')

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

size = (frame_width, frame_height)

result = cv2.VideoWriter('result.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result.write(frame)

    cv2.imshow('Video', frame)
    cv2.waitKey(10)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
