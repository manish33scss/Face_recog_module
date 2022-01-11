# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 02:16:12 2022

@author: ASUS
"""
import face_recognition
import cv2
import os
import pickle
import time
from trainer import TrainRecognizer
import json
from recognizer import Recognizer


print(cv2.__version__)



font=cv2.FONT_HERSHEY_SIMPLEX

results  = []
tr = TrainRecognizer()
r = Recognizer()

def json_landmarks(frame_counter, landmarks_list):
    for face_landmarks in landmarks_list:
        results.append({
           "FrameNo." : frame_counter,
            "Keypoints" : face_landmarks
    
        })
    with open("results.json", "w") as f:
        f.write(json.dumps(results))

if __name__ == "__main__":
    
    
    # for training images
    
    train = TrainRecognizer()
    #path to training images
    path = r"D:\Work\Codes\face_recogmition_api\demoImages-master\known"
    
    path_pkl = train.train(path)
    
    # after training enter the train.pkl path.
    
    #path_pkl = r"D:\Work\Codes\face_recogmition_api\train.pkl"
    cap = cv2.VideoCapture(r"D:\Work\Codes\face_recogmition_api\unknown\donald.gif")
    frame_counter = 0
    while True:
    
        _,frame=cap.read()
        frameSmall=cv2.resize(frame,(0,0),fx=.25,fy=.25)
        frameRGB=cv2.cvtColor(frameSmall,cv2.COLOR_BGR2RGB)
        

        
        #r = Recognizer.recognizer(frame)
        Names,Encodings, facePositions, allEncodings,landmarks = r.recognize(frameRGB, path_pkl )
        frame_counter += 1
        json_landmarks(frame_counter, landmarks)
        
        for (top,right,bottom,left),face_encoding in zip(facePositions,allEncodings):
            name='Unkown Person'
            matches=face_recognition.compare_faces(Encodings,face_encoding)
            if True in matches:
                first_match_index=matches.index(True)
                name=Names[first_match_index]
            top=top*4
            right=right*4
            bottom=bottom*4
            left=left*4
            cv2.rectangle(frame,(left,top),(right, bottom),(0,0,255),2)
            cv2.putText(frame,name,(left,top-6),font,.75,(0,0,255),2)
        cv2.imshow('Picture',frame)
        cv2.moveWindow('Picture',0,0)
        if cv2.waitKey(1)==ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
