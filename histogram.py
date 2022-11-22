from sklearn.cluster import KMeans
import utils
import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
import argparse
import timSort as sort

'''
    Compare the histograms of the fragments with the histograms of the images
    inputs:
        - pathToImage: path to the image
        - plot: if True, the histogram will be plotted
        - kmeans: if True, the image will be clustered
'''


def histogramGeneration(pathToImage, plot=False, kmeans=False):

    # load the image
    image = cv2.imread(pathToImage)

    if kmeans:
        # Convert the image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        a = np.asarray(image,dtype=np.float32)/255
        h, w, c = a.shape

        # reshape the image to be a list of pixels
        image = a.reshape((image.shape[0] * image.shape[1], 3))

        # cluster the pixel intensities
        clt = KMeans(n_clusters = 10)
        clt.fit(image)

        # build a histogram of clusters and then create a figure
        # representing the number of pixels labeled to each color

        # centroids and labels in clusted image
        centroids = clt.cluster_centers_
        labels = clt.labels_

        # centroids and labels in original image
        a2k = centroids[labels]
        a3k = a2k.reshape(h, w, c)

        # Convert the image to BGR
        image = cv2.cvtColor(a3k, cv2.COLOR_RGB2BGR)*255
    
    # Repetitions of the colors
    colorsRep = {}

    # Iterate over the pixels
    for i in range(len(image)):
        for j in range(len(image[i])):
            # If the color is in the dictionary, we add 1 to the repetitions
            hexa = utils.rgb_to_hex(round(image[i][j][0]), round(image[i][j][1]), round(image[i][j][2]))
            if hexa in colorsRep.keys():
                colorsRep[hexa] += 1
            else:
                colorsRep[hexa] = 1

    # Sort the dictionary
    l = list(colorsRep.values())
    sort.timSort(l)

    highers = l[-5:]

    # Get the colors of the 5 highest repetitions
    colorsFrag = []
    for key, value in colorsRep.items():
        for higher in highers:
            if value == higher:
                colorsFrag.append(key)
                break

    colorsFrag.reverse()

    # Plot the histogram
    if plot:
        ax = plt.axes()
        ax.set_facecolor("white")
        # plt.figure("Histogram in fragment")
        plt.xlabel("Colors")
        plt.ylabel("Repetitions")
        plt.title("Predominant colors in fragment")

        for i in range(len(colorsFrag)):
            plt.bar(colorsFrag[i], highers[i], 
                    color = [colorsFrag[i]])

        plt.show()

    return colorsFrag, highers, colorsRep



