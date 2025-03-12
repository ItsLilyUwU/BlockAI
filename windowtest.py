import mss
import cv2
import numpy as np

sct = mss.mss()
monitor = {"top": 0, "left": 200, "width": 980, "height": 770}  

while True:
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)

    # Show the captured area
    cv2.imshow("Tetris Capture Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()