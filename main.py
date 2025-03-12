import tensorflow as tf;
import mss as mss;
import numpy as np;
import cv2;
from tensorflow import keras

import keyboard as key
import time as t
import random as rd

print("TensorFlow version:", tf.__version__)

sct = mss.mss()
monitor = {"top": 0, "left": 200, "width": 980, "height": 770}

def preprocessImage(screenshot):
     img = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
     img = cv2.resize(img,(48,48))
     img = img.astype(np.float32) / 255.0
     img = np.expand_dims(img, axis=0)
     return img

def left(presses):
    for i in range(presses):
        key.press("left")
        t.sleep(0.03)
        key.release("left")
        t.sleep(0.03)

def right(presses):
    for i in range(presses):
        key.press("right")
        t.sleep(0.03)
        key.release("right")
        t.sleep(0.03)

def softDrop(time):
    key.press("down")
    t.sleep(time)
    key.release("down")
    t.sleep(0.03)

def turnA(presses):
    for i in range(presses):
        key.press("x")
        t.sleep(0.03)
        key.release("x")
        t.sleep(0.03)

def turnB(presses):
    for i in range(presses):
        key.press("z")
        t.sleep(0.03)
        key.release("z")
        t.sleep(0.03)

def beginGame():
    key.wait("t")
    for i in range(3):
        key.press("enter")
        t.sleep(0.06)
        key.release("enter")
        t.sleep(0.06)

print("Press T to begin Script")
beginGame()
t.sleep(0.8)
running = True

while True:
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)

    cv2.imshow("Tetris Capture Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    inputData = preprocessImage(frame)
    model = keras.models.load_model("tetris_model.h5")
    prediction = model.predict(inputData)
    blockType = np.argmax(prediction)
    print(blockType)
    match blockType:
        case 0:
            #I Block
            left(rd.randint(0,5))
        case 1:
            #J Block
            turnA(1)
            right(rd.randint(0,5))
        case 2:
            #L Block
            turnB(1)
            left(rd.randint(0,4))
        case 3:
            #O Block
            softDrop(1)
            right(1)
            left(rd.randint(0,2))
        case 4:
            #S Block
            turnA(1)
            right(rd.randint(0,3))
        case 5:
            #T Block
            turnB(rd.randint(1,3))
            left(2)
        case 6:
            #S Block
            turnB(1)
            left(rd.randint(0,3))
    right(rd.randint(0,1))
    left(rd.randint(0,1))
    turnA(rd.randint(0,2))
    softDrop(1)
