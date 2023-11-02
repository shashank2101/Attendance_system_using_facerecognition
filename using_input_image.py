import face_recognition
import numpy as np
import cv2
import os
from datetime import datetime
import csv
# Load the image containing a group of people
input_image = "s1.webp"

# Load known face encodings and names
known_face_encodings = []
known_face_names = []
known_face_files = ["rohit.jpg", "virat.jpg", "elon.jpg", "sundar.jpg","dhoni.jpg","rahul.jpeg"]

already_detected=[]
now=datetime.now()
current_date=now.strftime("%Y -%m-%d")
f=open(current_date+'.csv','w',newline= '')
lnwriter=csv.writer(f)
for file in known_face_files:
    image = face_recognition.load_image_file(file)
    encoding = face_recognition.face_encodings(image)[0]
    name = os.path.splitext(os.path.basename(file))[0]
    known_face_encodings.append(encoding)
    known_face_names.append(name)

# Load and preprocess the input image
input_image_data = face_recognition.load_image_file(input_image)
input_image_data = cv2.resize(input_image_data, (0, 0), fx=0.5, fy=0.5)
input_face_locations = face_recognition.face_locations(input_image_data,model='cnn')
input_face_encodings = face_recognition.face_encodings(input_image_data, input_face_locations)

# Perform face recognition on the loaded image
for face_encoding, face_location in zip(input_face_encodings, input_face_locations):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
    name = "Unknown"

    if True in matches:
        face_distance = face_recognition.face_distance(known_face_encodings, input_face_encodings)
        best_match_index = np.argmin(face_distance)
                #0            
        if matches[best_match_index]==1:
            name = known_face_names[best_match_index]
                    # print(name)
        if name in known_face_names and name not in already_detected:
            already_detected.append(name)
            current_time = now.strftime(" %D %I:%M %p")
            lnwriter.writerow([name, current_time,"Present"])
                
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    top, right, bottom, left = face_location

    # Draw a box around the face
    cv2.rectangle(input_image_data, (left, top), (right, bottom), (0, 0, 255), 2)

    # Draw a label with a name below the face
    cv2.rectangle(input_image_data, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(input_image_data, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

# Display the image with recognized faces
cv2.imshow("Recognized Faces", input_image_data)
cv2.waitKey(0)
cv2.destroyAllWindows()
