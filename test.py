import tensorflow as tf;
import mss as mss;
import numpy as np;
import cv2;

print("TensorFlow version:", tf.__version__)

sct = mss.mss()
monitor = {"top":100, "left":100 ,"width":400, "height":400}

while True:
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#hello = tf.constant('hello world')

#with tf.compat.v1.Session() as sess:
    #result = sess.run(hello)
    #print(result.decode())

