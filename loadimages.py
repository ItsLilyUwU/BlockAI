import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

#THIS IS A HELPER FUNCTION FOUND AT THIS LINK - https://pub.towardsai.net/how-to-create-a-new-custom-dataset-from-images-9b95977964ab
#Credit to Uday Sai and the staff at Medium for help on how to clean up our data
 
dataset_path = 'tetrisdataset.csv'
image_size=(48,48) 
 
def load():
    data = pd.read_csv(dataset_path)
    pixels = data['Pixels'].tolist()
    width, height= 48, 48
    gameplayScreenshots = []
    for pixel_sequence in pixels:
        screenshot = [int(pixel) for pixel in pixel_sequence.split(' ')]
        screenshot = np.asarray(screenshot).reshape(width, height,)
        a = screenshot
        screenshot = np.resize(screenshot.astype('uint8'),image_size)
        gameplayScreenshots.append(screenshot.astype('float32'))    
    gameplayScreenshots = np.asarray(gameplayScreenshots)
    A = gameplayScreenshots
    gameplayScreenshots = np.expand_dims(gameplayScreenshots, -1)
    return gameplayScreenshots, A
 
faces,A = load()
for item in A:
    plt.imshow(item.astype("uint8"))