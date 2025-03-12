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
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

csv = pd.read_csv("tetrisdataset.csv")
x = csv.drop(columns=["Labels"]) 
y = y = csv["Labels"].astype(np.int32)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

x = x["Pixels"].apply(lambda row: np.array(row.split(), dtype=np.float32)).tolist()
x = np.stack(x) / 255.0  
x = x.reshape(-1, 2304) 

x = np.array(x)

print(x.shape) 


#Splitting Data between Training, Validation, and Testing
for train_idx, val_idx in kfold.split(x, y):
    X_train, X_val = x[train_idx], x[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

def augment_data(x):
    noise = np.random.normal(0, 0.05, x.shape)
    return np.clip(x + noise, 0, 1)

X_train_augmented = np.vstack([X_train, augment_data(X_train)])
y_train_augmented = np.hstack([y_train, y_train])

print("Augmented Training Shape:", X_train_augmented.shape)
print("Augmented Labels Shape:", y_train_augmented.shape)

model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(2304,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(7, activation="softmax")
])


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005), 
    loss=keras.losses.SparseCategoricalCrossentropy(), 
    metrics=['accuracy']
)

print("Training Label Distribution:", Counter(y_train))
print("Validation Label Distribution:", Counter(y_val))

class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))
print("Computed Class Weights:", class_weights)

print("Unique Labels:", np.unique(y_train))
history = model.fit(
    X_train_augmented, y_train_augmented,  
    validation_data=(X_val, y_val), 
    epochs=100,  
    batch_size=8,  
    class_weight=class_weights
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
print("Validation Label Distribution:", Counter(y_val))
print(classification_report(y_val, y_pred, digits=4))
unique, counts = np.unique(y, return_counts=True)
print(dict(zip(unique, counts)))
