import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model


# Load dataset
test = pd.read_csv("test.csv")

X = test.drop("label", axis=1).values
y = test["label"].values


# Reshape images
X = X.reshape(-1,28,28,1) / 255.0


# Load trained model
model = load_model("best_sign_language_model.h5")


# Predict
predictions = model.predict(X)

predicted_labels = np.argmax(predictions, axis=1)


# Confusion matrix
cm = confusion_matrix(y, predicted_labels)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.show()