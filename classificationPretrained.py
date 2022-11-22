import tensorflow as tf
import numpy as np
import os
from keras.models import load_model
from PIL import Image, ImageOps
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from matplotlib import pyplot as plt

# Local variables
class_names = ['Rocas','Focas','Agua']
PATH_TO_FRAGMENTS = 'cropImages2/'
img_height = 224
img_width = 224
imagePath = PATH_TO_FRAGMENTS + "image29/"
PATH_TO_RESULTS = "ResultsPre-trained/results/"
PATH_TO_REVIEW = "ResultsPre-trained/review/"

'''
    Classifier for the pre-trained model
'''

def classifier():
    array = []
    # Load the model
    model = load_model('CNNPretrained.h5', compile=False)
    data = np.ndarray(shape=(1, img_height, img_width, 3), dtype=np.float32)
    count = 1

    for imageFragment in os.listdir(imagePath):
        path = imagePath + imageFragment
        img = cv2.imread(path)

        image = Image.open(path).convert('RGB')

        #resize the image to a 224x224 with the same strategy as in TM2:
        #resizing the image to be at least 224x224 and then cropping from the center
        size = (img_width, img_width)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        #turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Classification results
        if(class_name=="Focas" and confidence_score > 0.8):
            array.append(confidence_score)
            pathToResults = PATH_TO_RESULTS + imageFragment
            cv2.imwrite(pathToResults, img)
            print(imageFragment, "es una foca con un", 100* confidence_score, "% de certeza")

        if(class_name=="Focas" and (confidence_score > 0.5 and confidence_score < 0.8)):
            array.append(confidence_score)
            pathToReview = PATH_TO_REVIEW + imageFragment
            cv2.imwrite(pathToReview, img)
            print(imageFragment, "es una foca con un", 100* confidence_score, "% de certeza")
        count += 1

classifier()

'''
    Confusion matrix for the pre-trained model
    inputs:
        - modelToLoad: path to the model
        - folderOfImages: path to the folder with the images
        - classNames: names of the classes
        - height: height of the images
        - width: width of the images
        - NUMBER_AGUA: number of images of the class agua
        - NUMBER_FOCAS: number of images of the class focas
        - NUMBER_ROCAS: number of images of the class rocas
'''

def contructorOfCM(modelToLoad, folderOfImages, classNames, height, width, NUMBER_AGUA=10, NUMBER_FOCAS=15, NUMBER_ROCAS=20):
    
    # Load the model
    model = load_model(modelToLoad, compile=False)

    # variables to use
    countFocas = 0
    countRocas = 0
    countAgua = 0
    y_true = []
    y_pred = []
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Iterate over the folders
    for folder in os.listdir(folderOfImages):

        # Iterate over the images
        for image in os.listdir(folderOfImages + folder):

            # Load the image
            path = folderOfImages + folder + "/" + image
            image = Image.open(path).convert('RGB')
            size = (height, width)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            image_array = np.asarray(image)

            # Normalize the image
            normalized_image_array = (
                image_array.astype(np.float32) / 127.0) - 1
            data[0] = normalized_image_array

            # predict the class of the image
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = classNames[index]

            # Add the class to the array
            if folder == classNames[0]:
                y_true.append(classNames[0])
                y_pred.append(class_name)
                countAgua += 1
                if countAgua == NUMBER_ROCAS:
                    break
            elif folder == classNames[1]:
                y_true.append(classNames[1])
                y_pred.append(class_name)
                countFocas += 1
                if countFocas == NUMBER_FOCAS:
                    break
            elif folder == classNames[2]:
                y_true.append(classNames[2])
                y_pred.append(class_name)
                countRocas += 1
                if countRocas == NUMBER_AGUA:
                    break

    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classNames)
    accuracy_rocas = cm[0][0]/(NUMBER_ROCAS)
    accuracy_focas = cm[1][1]/(NUMBER_FOCAS)
    accuracy_agua = cm[2][2]/(NUMBER_AGUA)
    score = accuracy_score(y_true, y_pred)

    # Print the results
    print("Accuracy focas: ", accuracy_focas)
    print("Accuracy rocas: ", accuracy_rocas)
    print("Accuracy agua: ", accuracy_agua)
    print("Accuracy: ", score)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=classNames)
    disp.plot()
    plt.show()


contructorOfCM("CNNPretrained.h5", "testImages/",
               ["rocas", "focas", "agua"], 224, 224)