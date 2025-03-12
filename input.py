# Input.py

# I will be using the keyboard Python module to allow Python to control our keyboard.
# * Don't forget to install: pip3 install keyboard

import keyboard as key
import time as t
import random as rd

# Single Input: keyboard.send("space")
# Multi Input (+): keyboard.send("windows+d")
# Multistep Input (,): keyboard.send("alt+F4, space")

# These "send" inputs are so quick to the point our game will not be able to recognize
# the input. To fix this we will be using "press" and "release" with time.sleep delays.

running = False

def left(presses = 1):
    for i in range(presses):
        key.press("left")
        t.sleep(0.03)
        key.release("left")
        t.sleep(0.03)

def right(presses = 1):
    for i in range(presses):
        key.press("right")
        t.sleep(0.03)
        key.release("right")
        t.sleep(0.03)

def softDrop(time = 1):
    key.press("down")
    t.sleep(time)
    key.release("down")
    t.sleep(0.03)

def turnA(presses = 1):
    for i in range(presses):
        key.press("x")
        t.sleep(0.03)
        key.release("x")
        t.sleep(0.03)

def turnB(presses = 1):
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

beginGame()
t.sleep(0.8)
running = True
while(running): # This code just spams all commands at random, might have some funny results.
    turnA(rd.randint(0, 2))
    t.sleep(rd.randint(1, 2))
    left(rd.randint(0, 5))
    right(rd.randint(1, 4))
    t.sleep(rd.randint(1, 2))
    softDrop(2)
