import cv2
import os

dataPath = 'C:/Users/DiegoFdz/Documents/023_proyect/intentos/2/data/'
imagePaths = os.listdir(dataPath)

face_recognizer = cv2.face.EigenFaceRecognizer_create()

face_recognizer.read('modeloEigenFace.xml')

cap = cv2.VideoCapture('[video mp4]')

faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_defaul.xml')

while True:
    ret, frame = cap.read()
    if ret == False: break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces: 
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro, (150,150),interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        if result[1] <= 5700:
            cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x,y-25),1,1.3,(255,255,0),1,cv2.LINE_AA)    
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        else:
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(frame, '{}'.format(result), (x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
        

    cv2.imshow('frame',frame)
    k = cv2.waitKey(1)
    if k == 27: break

cap.release()
cv2.destroyAllWindows()