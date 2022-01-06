# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 01:28:06 2022

@author: ASUS
"""


import face_recognition
import cv2
import os
import pickle as pk




class TrainRecognizer:
    
    
 

    def __init__(self):
        pass
    
    def train(self,path):
        
        #image_dir=r'D:\Work\Codes\face_recogmition_api\demoImages-master\known'
        Encodings=[]
        Names=[]
        
        for root, dirs, files in os.walk(path):
            print(files)
            for file in files:
                path=os.path.join(root,file)
                
                name=os.path.splitext(file)[0]
                #print(name)
                person=face_recognition.load_image_file(path)
                encoding=face_recognition.face_encodings(person)[0]
                Encodings.append(encoding)
                Names.append(name)
        print(Names)
    
        with open('train.pkl','wb') as f:
            pk.dump(Names,f)
            pk.dump(Encodings, f)
            
        path = os.path.join(os.getcwd(),' train.pkl')
        return path
        
if __name__ == "__main__":
    
    train = TrainRecognizer()
    path = r"D:\Work\Codes\face_recogmition_api\demoImages-master\known"
    
    train.train(path)
       