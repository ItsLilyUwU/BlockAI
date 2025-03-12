import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras import callbacks
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


csv = pd.read_csv("tetrisdataset.csv")
x = csv.drop(columns=["Labels"]) 
y = y = csv["Labels"].astype(np.int32)

x = x["Pixels"].apply(lambda row: np.array(row.split(), dtype=np.float32)).tolist()
x = np.stack(x) / 255.0  # Normalize pixel values to [0,1]

print(x.shape) 


#Splitting Data between Training, Validation, and Testing
X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=42) 
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#Translating Data into TensorFlow Dataset
dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.shuffle(len(x)).batch(32)

model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(2304,)),  

    layers.Dense(32, activation="relu"),
     
    layers.Dense(16, activation="relu"),
    
    layers.Dense(8, activation="relu"),  
    layers.Dense(7, activation="softmax")  
])

model.compile(
    optimizer='adam',               
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']             
)
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,  
    validation_data=(X_val, y_val), 
    epochs=50,  
    batch_size=32,  
)

y_pred_probs = model.predict(X_val)  # Get probability predictions
y_pred = np.argmax(y_pred_probs, axis=1) 

cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(7), yticklabels=range(7))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print(classification_report(y_val, y_pred, digits=4))