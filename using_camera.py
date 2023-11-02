import face_recognition
import numpy as np
import os
import csv
import glob
from datetime import datetime
import cv2
import time
video_capture=cv2.VideoCapture(0)
known_face_encoding=[]
known_face_names=[]
# to load images from a folder
for filename in glob.glob("known_faces/*"):
    print("Processing:", filename)

    # Load the image    
    face = face_recognition.load_image_file(filename)

    # Check if any face was detected
    face_encodings = face_recognition.face_encodings(face)
    if len(face_encodings) > 0:
        # If a face was detected, append the encoding and the corresponding name
        known_face_encoding.append(face_encodings[0])
        known_face_names.append(os.path.basename(filename).split(".")[0])
    else:
        print("No face found in", filename)



rohit_image=face_recognition.load_image_file("rohit.jpg")
ri=face_recognition.face_encodings(rohit_image)[0]

# elon_image=face_recognition.load_image_file("elon.jpg")
# ei=face_recognition.face_encodings(elon_image)[0]

# sundar_image=face_recognition.load_image_file("sundar.jpg")
# si=face_recognition.face_encodings(sundar_image)[0]

# virat_image=face_recognition.load_image_file("virat.jpg")
# vi=face_recognition.face_encodings(virat_image)[0]
me_image=face_recognition.load_image_file("exam.jpg")
me=face_recognition.face_encodings(me_image)[0]
# known_face_encoding=[ri,vi,ei,si,me]
# known_face_names=["rohit","virat","elon","sundar","Me"]
print(known_face_names)
students=known_face_names.copy()
# message_name=None
face_locations=[]
face_encodings=[]
face_names=[]
s=True
b=False
# global message_name
# message_name=""
already_detected=[]
now=datetime.now()
current_date=now.strftime("%Y -%m-%d")
f=open(current_date+'.csv','w',newline= '')
lnwriter=csv.writer(f)
def capture_image():
    # while True:
        global message_name
        message_name=""
        _,frame = video_capture.read()
        # frame=cv2.imread("group.jpg")
        small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
        if s:
            face_locations=face_recognition.face_locations(rgb_small_frame,model='cnn')
    
            face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)  
            face_names = []
            for face_encoding in face_encodings:
                name = ""
                matches = face_recognition.compare_faces(known_face_encoding, face_encoding,tolerance=0.4)
    #matches=[1,0,0,1]
    #fac_dist=[10,20,30,40]

                # face_encodings=
                face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
                best_match_index = np.argmin(face_distance)
                #0            
                if matches[best_match_index]==1:
                    name = known_face_names[best_match_index]
                    # print(name)
                if name in known_face_names and name not in already_detected:
                    already_detected.append(name)
                    current_time = now.strftime(" %D %I:%M %p")
                    lnwriter.writerow([name, current_time,"Present"])
                    # already_detected.append(name)
                    message_name = name 
                    message_display_start = time.time() 
                # if message_name!="" and message_display_start is not None:
                #     elapsed_time = time.time() - message_display_start
                #     if elapsed_time <= 9: 
                #         cv2.putText(frame, f" {message_name} : Present", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                #     else:
                #         message_name = ""  # Clear the message after 2 seconds
                #         message_display_start = None

        # cv2.namedWindow("attenda nce system", cv2.WINDOW_NORMAL)    
        # cv2.imshow("attendance system",frame)
        # if cv2.waitKey(1) & 0xFF==32:
        #     break
capture_image()
print(set(already_detected))
video_capture.release()
cv2.destroyAllWindows()
f.close()

