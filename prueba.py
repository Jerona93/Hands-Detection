import cv2

captura=cv2.VideoCapture(0)

while True:
    ret,imagen=captura.read()
    if ret == True:
        cv2.imshow('video',imagen)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
captura.release()
cv2.destroyAllWindows()