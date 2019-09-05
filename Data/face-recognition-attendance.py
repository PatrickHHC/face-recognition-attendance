# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 10:34:32 2019

@author: 11431
"""


import face_recognition
import pandas as pd
import cv2
from time import localtime,strftime
import glob2 as gb
import pymysql
from sqlalchemy import create_engine
from pandas.core.frame import DataFrame
import dlib
from keras.models import load_model
import numpy as np
import datetime
from PIL import Image, ImageDraw



def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    draw.text((left, top), text, textColor)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=3, minSize=(40, 40))
    color = (255, 0, 0)

    for (x, y, w, h) in faces:
        gray_face = gray[(y):(y + h), (x):(x + w)]
        gray_face = cv2.resize(gray_face, (48, 48))
        gray_face = gray_face / 255.0
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        


emotion_classifier = load_model(r'D:\face-recognition-attendance\simple_CNN.530-0.65.hdf5')
emotion_labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'calm'
}
face_detector = cv2.CascadeClassifier(r"D:\face-recognition-attendance\haarcascade_frontalface_default.xml")
# 操作数据库
engine = create_engine("mysql+pymysql://root:localhost@localhost:3306/Data?charset=utf8")

video_path = 'http://patrick:hhc@172.27.12.177:8081/video'
if len(str(video_path)) == 1:
    vid = cv2.VideoCapture(int(video_path))
else:
    vid = cv2.VideoCapture(video_path)
print('start read video')
print(vid)
if not vid.isOpened():
    raise IOError("Couldn't open webcam or video")
        
        
img_path=gb.glob(r'D:\Working\photo\\*.jpg')
known_face_names=[]
known_face_encodings=[]

        
for i in img_path:
    picture_name=i.replace('D:\Working\photo\\*.jpg','')
    picture_newname=picture_name.replace('.jpg','')
    picture_newname=picture_newname.replace('D:\Working\photo\\','')
    someone_img = face_recognition.load_image_file(i)
    someone_face_encoding = face_recognition.face_encodings(someone_img)[0]
    known_face_names.append(picture_newname)
    known_face_encodings.append(someone_face_encoding)
    someone_img=[]
    someone_face_encoding=[]

	
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

 
while True:
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=3, minSize=(40, 40))
    color = (255, 0, 0)

    for (x, y, w, h) in faces:
        gray_face = gray[(y):(y + h), (x):(x + w)]
        gray_face = cv2.resize(gray_face, (48, 48))
        gray_face = gray_face / 255.0
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
        emotion = emotion_labels[emotion_label_arg]
        cv2.rectangle(frame, (x + 10, y + 10), (x + h - 10, y + w - 10), (255, 255, 255), 2)
        frame = cv2ImgAddText(frame, emotion, x + h * 0.3, y, color, 20)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame=small_frame[:,:,::-1]

    
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for i in face_encodings:
                match = face_recognition.compare_faces(known_face_encodings,i,tolerance=0.39)
                if True in match:
                    match_index=match.index(True)
                    name = "match"
                #To print name and time and emotion
                    cute_clock =strftime("%Y-%m-%d %H:%M:%S",localtime())                                        
                    data={"name" : known_face_names[match_index],
                       "time" : str(cute_clock),
                       "emotion" : emotion_labels[emotion_label_arg]}
                    data=DataFrame(data,index = [0])                    
                    pd.io.sql.to_sql(frame=data,name = 'timecard',con = engine, schema='Data',if_exists = 'append',index = False,index_label = False)
                    print (data)
                else:
                    name = "unknown"
                face_names.append(name)

    process_this_frame = not process_this_frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255),  2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left+6, bottom-6), font, 1.0, (255, 255, 255), 1)
    
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()


if __name__ == '__main__':
    tic = datetime.datetime.now()
    toc = datetime.datetime.now()
    print(toc - tic)