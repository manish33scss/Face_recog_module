# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 01:28:14 2022

@author: ASUS
"""

import face_recognition
import cv2
import os
import pickle
#import time
#print(cv2.__version__)

Encodings=[]
Names=[]







class Recognizer:
    
    

    def __init__(self):
        self.Encodings=list()
        self.Names=list()
    
    
    def recognize(self, frame, data_path):
        with open(data_path,'rb') as f:
            self.Names=pickle.load(f)
            self.Encodings=pickle.load(f)
        facePositions=face_recognition.face_locations(frame,model='cnn')
        allEncodings=face_recognition.face_encodings(frame,facePositions)
            
        return self.Names, self.Encodings, facePositions, allEncodings