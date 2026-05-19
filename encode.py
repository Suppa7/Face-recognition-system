import cv2
import face_recognition
import pickle
import os

# 1. วนลูปเข้าไปในแต่ละโฟลเดอร์ของบุคคล
folder_images = 'Images'
img_list = []
student_id = []

# วนลูปอ่านชื่อโฟลเดอร์ (ซึ่งก็คือ ID หรือชื่อของคน)
for person_name in os.listdir(folder_images):
    person_folder_path = os.path.join(folder_images, person_name)

    # ตรวจสอบว่าเป็นโฟลเดอร์จริงๆ ไม่ใช่ไฟล์อื่น
    if not os.path.isdir(person_folder_path):
        continue

    # วนลูปอ่านไฟล์รูปภาพทั้งหมดในโฟลเดอร์ของคนนั้นๆ
    for filename in os.listdir(person_folder_path):
        image_path = os.path.join(person_folder_path, filename)
        
        # อ่านไฟล์รูปภาพ
        img = cv2.imread(image_path)
        
        # ตรวจสอบว่าอ่านรูปได้สำเร็จหรือไม่
        if img is not None:
            img_list.append(img)
            student_id.append(person_name)  # ใช้ชื่อโฟลเดอร์เป็น ID
        else:
            print(f"Warning: Could not read image {image_path}")

print(f"Loaded {len(img_list)} images from {len(os.listdir(folder_images))} people.")
print("Encoding faces... This might take a while.")

# 2. ฟังก์ชันสร้าง Encoding (ไม่ต้องแก้ไข)
def find_encoding(imageslist):
    encode_list = []
    for img in imageslist:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # ใช้ face_encodings แบบที่อาจคืนค่าเป็นลิสต์ว่างได้ เพื่อป้องกัน error
        encodes = face_recognition.face_encodings(img)
        if encodes: # ตรวจสอบว่าเจอใบหน้าในรูปหรือไม่
            encode_list.append(encodes[0])
        else:
            print("Warning: No face found in one of the images, skipping.")
            
    return encode_list

# 3. สร้างและบันทึก Encoding (ไม่ต้องแก้ไข)
encode_list_known = find_encoding(img_list)
encode_list_known_with_ids = [encode_list_known, student_id]

# ตรวจสอบว่ามีใบหน้าที่ถูก encode อย่างน้อยหนึ่งใบหน้าก่อนบันทึก
if encode_list_known:
    with open("EncodeFile3.p", 'wb') as file:
        pickle.dump(encode_list_known_with_ids, file)
    print("Encoding complete and file 'EncodeFile.p' has been saved.")
else:
    print("Could not encode any faces. 'EncodeFile.p' was not created.")