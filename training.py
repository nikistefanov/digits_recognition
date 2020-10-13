import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D

###
path = 'digits_dataset'
testRatio = 0.2
validationRatio = 0.2
imageDimensions = (32, 32, 3)
batchSize = 50
epochs = 10
stepsPerEpoch = 100
###
images = []
classNumber = []
dataList = os.listdir(path)
classesLenght = len(dataList)

# Train application
## Import the dataset of images to be trained
print("Importing dataset.")
### Iterate throught all data image folders
for folder in range (0, classesLenght):
    currentFilePath = path + "/" + str(folder)
    currentImgFolder = os.listdir(currentFilePath)

    ### Iterate throught all the images inside current folder
    for img in currentImgFolder:
        currentImg = cv2.imread(currentFilePath + "/" + str(img))
        ### Reduce the size to be performance friendly
        currentImg = cv2.resize(currentImg, (imageDimensions[0], imageDimensions[1]))
        ### Store image and coresponding label
        images.append(currentImg)
        classNumber.append(folder)
    print(folder, end=" ", flush=True)
    
print("")
print("Imported: ", len(classNumber))

## Convert images to numpy array
images = np.array(images)
classNumber = np.array(classNumber)
print(images.shape)

## Split the data
### This will shuffle the dateset evenly in order to train and test it right
### testRation = 0.2 means: 80% training and 20% testing
### X_train contains all the images
### y_train contains all the labels for each image
X_train, X_test, y_train, y_test = train_test_split(images, classNumber, test_size = testRatio)

### split again for validation purposes
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size = validationRatio)

print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

numOfSamples = []
for x in range(0, classesLenght):
    numOfSamples.append(len(np.where(y_train == x)[0]))

print(numOfSamples)

# Create bar chart
# plt.figure(figsize=(10, 5))
# plt.bar(range(0, classesLenght), numOfSamples)
# plt.title("Number of Images for each label")
# plt.xlabel("Label")
# plt.ylabel("Number of images")
# plt.show()

# Preprocess image
def preProcessing(img):
    ### convert it to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ### distrube the light of the image equaly
    img = cv2.equalizeHist(img)
    ### normalize value from 0-255 to 0-1. Better for training process
    img = img/255

    return img

### map every image in train, test and validation
X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))

### show image
# img = X_train[30]
# img = cv2.resize(img, (300, 300))
# cv2.imshow("PreProcessed", img)
# cv2.waitKey(0)

## Add depth to images
### The first 3 params will remain the same
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

## Augment images (translation, zoom, rotaion, shift) - will make the dateset more generic
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)

y_train = to_categorical(y_train, classesLenght)
y_test = to_categorical(y_test, classesLenght)
y_validation = to_categorical(y_validation, classesLenght)

def myModel():
    numOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    numOfNode = 500

    model = Sequential()
    model.add((Conv2D(numOfFilters,
                      sizeOfFilter1,
                      input_shape = (imageDimensions[0], imageDimensions[1],1),
                      activation = 'relu')))
    model.add((Conv2D(numOfFilters, sizeOfFilter1, activation = 'relu')))
    model.add(MaxPooling2D(pool_size = sizeOfPool))

    model.add((Conv2D(numOfFilters / 2, sizeOfFilter2, activation = 'relu')))
    model.add((Conv2D(numOfFilters / 2, sizeOfFilter2, activation = 'relu')))
    model.add(MaxPooling2D(pool_size = sizeOfPool))
    model.add(Dropout(0.5)) #reduce orphaning

    model.add(Flatten())
    model.add(Dense(numOfNode, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classesLenght, activation = 'softmax'))

    model.compile(Adam(lr = 0.001),
                       loss = 'categorical_crossentropy',
                       metrics = ['accuracy'])

    return model

# Start the training
model = myModel()
history = model.fit_generator(dataGen.flow(X_train, y_train, batch_size = batchSize),
                    steps_per_epoch = stepsPerEpoch,
                    epochs = epochs,
                    validation_data = (X_validation, y_validation),
                    shuffle = 1)

### how the variation of loss and the variation of accuracy
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')

## uncomment in order to see the loss/accuracy graphs
#plt.show()

score = model.evaluate(X_test, y_test, verbose = 0)
print('Test score: ', score[0])
print('Test accuracy: ', score[1])

model.save("model_trained.h5")