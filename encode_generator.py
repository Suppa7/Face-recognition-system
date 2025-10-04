import cv2
import face_recognition
import pickle
import os

#import student images
folder_images = 'Images'
path_list = os.listdir(folder_images)
img_list = []
student_id = []
for path in path_list:
    img_list.append(cv2.imread(os.path.join(folder_images,path)))
    student_id.append(os.path.splitext(path))

def find_encoding(imageslist):
    encode_list = []
    for img in imageslist:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)

    return encode_list

encode_list_known = find_encoding(img_list)
encode_list_known_id = [encode_list_known, student_id]

file = open("EncodeFile.p",'wb')
pickle.dump(encode_list_known_id,file)
file.close()