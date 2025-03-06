import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split

csv = pd.read_csv("tetrisdataset.csv")

csv.drop(columns=["target_column"])

X = csv.drop(columns=["target_column"])
y = csv["target_column"]

#Splitting Data between Training, Validation, and Testing
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42) 
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#Translating Data into TensorFlow Dataset
dataset = csv.data.Dataset.from_tensor_slices((X.values, y.values))
dataset = dataset.shuffle(len(X)).batch(32)

model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(2,)),  
    layers.Dense(32, activation="relu"),  
    layers.Dense(16, activation="relu"),  
    layers.Dense(1, activation="softmax")  
])
