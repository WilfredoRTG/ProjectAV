
# Detection of pinnipeds

## Authors

- Mauricio Hoyos Hernandez 166468
- Javier Alejandro Villegas Zamora 168165
- Wilfredo Rafael Tablero Gomez 166918

## Class Data

- LIS4042 Visión Artificial
- Profesora Zobeida Guzmán
- Otoño 2022
- UDLAP

## Content description

The folder structure of this repository it is correspondent of 7 folders.

| Folder Name        | Description                                      |
| ------------------ | -------------------------------------------------|
| CropImages         | Croped images of our dataset by 500 x 500 pixels |
| CropImages2        | Croped images of our dataset by 125 x 125 pixels |
| dataSet            | Images of 3992 x 2992 pixels                     |
| eliminatedImages   | Eliminated images of our umbralization           |
| ResultsPre-trained | Results of the pretrained CNN                    |
| templateImages     | Image templates to made de comparision in SIFT   |
| testImages         | Test images to made the confusion matrix         |


In the repository we have 10 files.

| File Name                | Extension | Description                                                     |
| ------------------------ |-----------| --------------------------------------------------------------- |
| CNNPretrained            | .h5       | Pretrained model                                                |
| umbral                   | .npy      | Array of numpy with all the color umbralization                 |
| SIFT                     | .py       | Algorithm to made the unsupervised learning technique           |
| classificationPretrained | .py       | Algorithm to made the supervised learning technique             |
| cropImage                | .py       | Algorithm to crop the images by 500x500 and 125x125             |
| eliminationOfFragmet     | .py       | Algorithm to eliminate images according to the umbral of colors |
| histogram                | .py       | Histogram generation                                            |
| timsort                  | .py       | Sorting algorithm TimSort                                       |
| umbralGeneration         | .py       | Generation of umbral of colors                                  |
| utils                    | .py       | Utilities to some functions                                     |

- To execute the code:
    - To visualizate the results of the unsupervised learning technique (SIFT), execute the SIFT script
    - In order to visualizate the results of the supervised learning technique (CNN), execute the classificationPretrained script

## Requerimientos

### Programming language

- Python 3.9

### Libraries

- Matplotlib
- Keras
- Tensorflow
- OpenCV
- Numpy
- Sckit Learn

## Distribuition of tasks

| Stage               | Preposition executet in final project                  | Developed by      | Evaluated or integrated by                        |
| ------------------  | ------------------------------------------------------ | ----------------- | ------------------------------------------------- |
| Preprocessing       | Gray scale, sharpening, histogram equalization filters | Wilfredo Tablero  | Mauricio Hoyos, Javier Villegas                   |
| Features extraction | SIFT                                                   | Mauricio Hoyos    | Javier Villegas, Wilfredo Tablero                 |
| Classification      | Template matching using SIFT                           | Mauricio Hoyos    | Javier Villegas                                   |
| Evaluation          | Confusion matrix and accuracy metrics                  | Wilfredo Tablero  | Mauricio Hoyos                                    |
| Comparision         | SIFT template matching and pretrained CNN              | Mauricio Hoyos    | Javier Villegas, Wilfredo Tablero, Mauricio Hoyos |
