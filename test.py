import tensorflow as tf;
import mss as mss;
import numpy as np;
import cv2;

print("TensorFlow version:", tf.__version__)

sct = mss.mss()
monitor = {"top":100, "left":100 ,"width":400, "height":400}

while True:
    screenshot = sct.grab(monitor)
    transcriptedScreenshot = np.array(screenshot)
    transcriptedScreenshot = cv2.cvtColor(transcriptedScreenshot, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Tetris Captures",transcriptedScreenshot)
    if cv2.waitKey == ord("q"):
        break



#hello = tf.constant('hello world')

#with tf.compat.v1.Session() as sess:
    #result = sess.run(hello)
    #print(result.decode())

