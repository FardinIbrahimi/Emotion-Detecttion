from keras.models import load_model
from time import sleep
from keras.utils.image_utils import img_to_array
import cv2
import numpy as np
import streamlit as st
from datetime import datetime


st.title("FEELY version 10.15.22")
st.write("Detects Emotions Intelligently!")
st.sidebar.title("FEELY Values Feelings!")
startButton = st.sidebar.button("Run Feely")
StopButton = st.sidebar.button("Stop Feely")


face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier =load_model('model.h5')

emotion_labels = ['قهر','گیچ','ترسیده','خوشحال','عادی', 'غمگین', 'متعجب']
startButton = True
if startButton and startButton == True:
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                prediction = classifier.predict(roi)[0]
                label=emotion_labels[prediction.argmax()]
                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                st.sidebar.write(datetime.now(),f"{label}") 
                sleep(1)
                if label == "غمگین":
                    st.write("شما غمگین هستید")
                elif  label == "عادی":
                    st.write("وضعیت چهره تان عادی است")
                elif label == "قهر":
                    st.write("شما قهر هستید")
                elif label == "متعجب":
                    st.write ("شما را چیزی متحیر ساخته است")
                elif label == "ترسیده":
                    st.write("شما از چیزی ترسیده اید")
                elif label == "خوشحال":
                    st.write("شما خوشحال هستید")
                else:
                    st.write("شما گیچ شده اید")                    
            else:
                cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                
        cv2.imshow('Emotion Detector',frame)
        if StopButton:
            startButton = False
            break
        
    cap.release()
    cv2.destroyAllWindows()