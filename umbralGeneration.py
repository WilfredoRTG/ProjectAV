from histogram import histogramGeneration
import os 
from numpy import save

PATH_TO_WATER = 'testImages/agua/'
colorUmbral = {}
umbral = []

for image in os.listdir(PATH_TO_WATER):
    # histogram generation
    colorsFrag = histogramGeneration(PATH_TO_WATER + image, kmeans=True)[0]
    
    # umbral generation
    for i in range(len(colorsFrag)):
        # if the color is in the umbral, we add it to the repetitions dictionary
        if colorsFrag[i] in colorUmbral.keys():
            colorUmbral[colorsFrag[i]] += 1
        else:
            colorUmbral[colorsFrag[i]] = 1

    umbral = list(colorUmbral.keys())

# save the umbral
save('umbral.npy', umbral)