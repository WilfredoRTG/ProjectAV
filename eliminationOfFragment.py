from histogram import histogramGeneration
from numpy import load
import os
from timSort import timSort
import shutil

PATH_TO_CROP = 'cropImages/'
umbral = load('umbral.npy')

image = "image29/"

for imageFragment in os.listdir(PATH_TO_CROP + image):
    # variables to use
    repetitions = {}

    # generation of the histogram
    colorsFrag = histogramGeneration(PATH_TO_CROP + image + "/" + imageFragment, kmeans=True)[0]

    # comparison of the histograms
    for i in range(len(colorsFrag)):
        if colorsFrag[i] in umbral:
            # if the color is in the umbral, we add it to the repetitions dictionary
            print(imageFragment, "eliminar")
            src_path = PATH_TO_CROP + image + "/" + imageFragment 
            dst_path = "eliminatedImages/" + "image31" + imageFragment
            shutil.move(src_path, dst_path)