

import cv2
import numpy as np
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from keras.models import load_model




model = load_model("Models/age_gender_model.keras")

model.summary()




gender_dict = {0: 'Male', 1: 'Female'}




# Open webcam (index 0 for the default webcam)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Resize the frame to 128x128 and convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (128, 128))
    reshaped_frame = resized_frame.reshape(1, 128, 128, 1)  # Model input shape

    # Normalize the image to [0, 1] (same preprocessing as during training)
    reshaped_frame = reshaped_frame / 255.0

    # Make predictions on the current frame
    pred = model.predict(reshaped_frame)

    # Extract gender and age predictions
    pred_gender = gender_dict[round(pred[0][0][0])]
    pred_age = round(pred[1][0][0])

    # Display the predictions on the frame
    cv2.putText(frame, f"Gender: {pred_gender}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Age: {pred_age}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow("Live Gender and Age Prediction", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()






