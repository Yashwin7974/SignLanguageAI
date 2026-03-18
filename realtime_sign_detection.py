import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("best_sign_language_model.h5")

# Labels from Sign Language MNIST (J and Z not included)
labels = [
'A','B','C','D','E','F','G','H','I',
'K','L','M','N','O','P','Q','R','S',
'T','U','V','W','X','Y'
]

cap = cv2.VideoCapture(0)

prediction = ""
last_prediction_time = 0
prediction_delay = 0.7  # seconds between predictions

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    # Region of Interest
    roi = frame[100:400, 100:400]

    # Predict only after delay
    if time.time() - last_prediction_time > prediction_delay:

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (28, 28))

        img = gray.reshape(1, 28, 28, 1) / 255.0

        pred = model.predict(img, verbose=0)
        prediction = labels[np.argmax(pred)]

        last_prediction_time = time.time()

    # Draw ROI box
    cv2.rectangle(frame, (100,100), (400,400), (0,255,0), 2)

    # Show prediction
    cv2.putText(frame,
                prediction,
                (100,90),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0,255,0),
                2)

    cv2.imshow("Sign Detection - Press Q to Quit", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        print("Closing camera...")
        break

cap.release()
cv2.destroyAllWindows()