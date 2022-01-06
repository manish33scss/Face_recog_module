import face_recognition
import cv2
import os
import pickle
import time
from trainer import TrainRecognizer

from recognizer import Recognizer
from flask import Flask, render_template, request, Response


#Simple flask app for face recognition in browser.

app = Flask(__name__)

r = Recognizer()
font=cv2.FONT_HERSHEY_SIMPLEX
#%% app route
@app.route('/')
def upload_file():
   return render_template('index.html')



def recognizer():
    path_pkl = r"D:\Work\Codes\face_recogmition_api\train.pkl"
    cap = cv2.VideoCapture(r"D:\Work\Codes\face_recogmition_api\unknown\donald.gif")
    
    while True:
    
        _,frame=cap.read()
        frameSmall=cv2.resize(frame,(0,0),fx=.25,fy=.25)
        frameRGB=cv2.cvtColor(frameSmall,cv2.COLOR_BGR2RGB)
        
        #class
    # =============================================================================
    #     facePositions=face_recognition.face_locations(frameRGB,model='cnn')
    #     allEncodings=face_recognition.face_encodings(frameRGB,facePositions)
    # =============================================================================
        
        #r = Recognizer.recognizer(frame)
        Names,Encodings, facePositions, allEncodings = r.recognize(frameRGB, path_pkl )
        
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
            frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY,20])[1].tobytes()
            time.sleep(0.015)
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
            

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(recognizer(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

    

#%% Main
if __name__ == '__main__':
    app.run(port = '8080')
