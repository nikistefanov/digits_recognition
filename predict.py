import numpy as np
import cv2
from keras.models import load_model

###
width = 640
height = 480
threshold = 0.8
###

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

model = load_model("model_trained.h5")

# Preprocess image
def preProcessing(img):
    ### convert it to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ### distrube the light of the image equaly
    img = cv2.equalizeHist(img)
    ### normalize value from 0-255 to 0-1. Better for training process
    img = img/255

    return img

while True:
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preProcessing(img)
    cv2.imshow("Processed image: ", img)
    img = img.reshape(1, 32, 32, 1)

    # Predict
    classIndex = int(model.predict_classes(img))
    predictions = model.predict(img)
    probVal = np.amax(predictions)
    print(classIndex, probVal)

    ## if the prediction is greater then the threshold show the digit
    if probVal > threshold:
        cv2.putText(imgOriginal,
                    str(classIndex) + ": " + str(probVal),
                    (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 0, 255),
                    1)

    cv2.imshow("Original image", imgOriginal)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break
