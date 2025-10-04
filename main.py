import cv2
import os
import pickle
import face_recognition
import numpy as np
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from datetime import datetime

cred = credentials.Certificate("service_firebase.json")
firebase_admin.initialize_app(cred,{
    'databaseURL' : "https://face-recognition-project-d959a-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

#set up camera
video = cv2.VideoCapture(0)

#set up frame's design
img_background = cv2.imread("Resources/Face_cognition_bg.png")
folder_mode = 'Resources/Modes'
mode_list = os.listdir(folder_mode)
img_mode_list = []
for path in mode_list:
    img_mode_list.append(cv2.imread(os.path.join(folder_mode,path)))

#load encode file
file = open('EncodeFile.p','rb')
encode_list_known_id = pickle.load(file)
file.close()
encode_list_known, student_id = encode_list_known_id

modetype = 0
counter = 0
id = -1

while True:
    ret, frame = video.read()

    img_resize = cv2.resize(frame,(0,0),None, 0.25, 0.25)
    img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)

    face_cur_frame = face_recognition.face_locations(img_resize)
    encode_cur_frame = face_recognition.face_encodings(img_resize, face_cur_frame)


    img_background[168:168+480, 72:72+640] = frame
    img_background[168:168+480, 753:753+375] = img_mode_list[modetype]

    if face_cur_frame:
        for encode_face, face_loc in zip(encode_cur_frame, face_cur_frame):
            tolerance = 0.45
            matches = face_recognition.compare_faces(encode_list_known,encode_face, tolerance)
            face_distance = face_recognition.face_distance(encode_list_known,encode_face)

            match_index = np.argmin(face_distance)

            if matches[match_index] and face_distance[match_index] < tolerance:
                #print(student_id[match_index])
                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                bbox = 55 + x1, 162 + y1, x2-x1, y2-y1
                img_background = cvzone.cornerRect(img_background,bbox,rt=0)
                id = student_id[match_index][0]
                
                if counter ==0 :
                    counter = 1
                    modetype = 1
            else:
                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                bbox = 55 + x1, 162 + y1, x2-x1, y2-y1
                img_background = cvzone.cornerRect(img_background, bbox, rt=0)
                cv2.putText(img_background, "Unknown", (bbox[0], bbox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            if counter != 0 :
                
                if counter ==1:
                    #get data
                    student_info = db.reference(f'Students/{id}').get()
                    #update data
                    datetime_object = datetime.strptime(student_info['last_attendance_time'],
                                                    "%Y-%m-%d %H:%M:%S")
                    
                    if datetime_object.date() != datetime.now().date():
                        ref = db.reference(f'Students/{id}')
                        student_info['total_attendance'] += 1
                        ref.child('total_attendance').set(student_info['total_attendance'])
                        ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    else:
                        modetype = 3
                        counter = 0

                if modetype != 3:
                    if 10<counter<20:
                        modetype = 2
                    
                    if counter<=10:
                        cv2.putText(img_background,str(student_info['total_attendance']),(1015,544),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
                        cv2.putText(img_background,str(student_info['ID']),(900,423),
                                    cv2.FONT_HERSHEY_COMPLEX,0.4,(255,255,255),1)
                        cv2.putText(img_background,str(student_info['Name']),(850,485),
                                    cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
                    
                    counter+=1

                    if counter>=20:
                        counter = 0
                        modetype = 0
    else:
        modetype = 0
        counter = 0

    cv2.imshow("face cognition", img_background)
    k = cv2.waitKey(1)
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()
  