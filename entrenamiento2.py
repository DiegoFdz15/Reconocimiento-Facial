import cv2
import os
import numpy as np

dataPath = 'C:/Users/DiegoFdz/Documents/023_proyect/intentos/2/data/'

peopleList = os.listdir(dataPath)
labels = []
facesData = []

#print('Lista: ', peopleList)

label = 0

for nameDir in peopleList:
    personPath = dataPath + nameDir
    print("Leyendo imagenes...")

    for filename in os.listdir(personPath):
        #print('Rostros: ', nameDir + '/' + filename)
        labels.append(label)
        facesData.append(cv2.imread(personPath+'/'+filename,0))
        image = cv2.imread(personPath+'/'+filename,0)
        #cv2.imshow('image',image)
        #cv2.waitKey(10)
    label = label + 1

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))

# Almacenamiento
face_recognizer.write('modeloLBPHFace.xml')
print("Almacenamiento completo.")