import cv2
import os
import glob
import albumentations as A
import numpy as np

# --- 1. กำหนดค่าต่างๆ ---

# โฟลเดอร์ที่เก็บรูปต้นฉบับ
INPUT_FOLDER = "source_faces" 
# โฟลเดอร์สำหรับเก็บรูปที่สร้างขึ้นใหม่ (ตามที่คุณต้องการ)
OUTPUT_FOLDER = "Images"
# จำนวนรูปที่ต้องการสร้างจากรูปต้นฉบับ 1 รูป
IMAGES_TO_GENERATE_PER_SOURCE = 10 

# --- 2. สร้าง Pipeline การ Augmentation ---
# คุณสามารถเพิ่ม/ลด หรือปรับค่าการ Augmentation ได้ตามต้องการ
# p คือความน่าจะเป็นที่จะใช้เทคนิคนั้นๆ (เช่น p=0.5 คือมีโอกาส 50%)
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2) # ปรับปรุง contrast ในพื้นที่เล็กๆ
])

# --- 3. เตรียมโฟลเดอร์ ---
# สร้างโฟลเดอร์ INPUT และ OUTPUT หากยังไม่มี
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"เตรียมพร้อม! กรุณานำรูปภาพใบหน้าต้นฉบับไปใส่ในโฟลเดอร์ '{INPUT_FOLDER}'")
print("---------------------------------------------------------")

# --- 4. เริ่มกระบวนการสร้างรูปภาพ ---

# ค้นหารูปภาพทั้งหมดในโฟลเดอร์ INPUT (รองรับ .jpg, .jpeg, .png)
image_paths = glob.glob(os.path.join(INPUT_FOLDER, '*.[jJ][pP][gG]')) \
            + glob.glob(os.path.join(INPUT_FOLDER, '*.[pP][nN][gG]'))

if not image_paths:
    print(f"ไม่พบรูปภาพในโฟลเดอร์ '{INPUT_FOLDER}' กรุณาตรวจสอบและลองอีกครั้ง")
else:
    print(f"พบรูปภาพต้นฉบับ {len(image_paths)} รูป, เริ่มสร้างรูปภาพ...")

    total_generated = 0
    # วนลูปสำหรับแต่ละรูปภาพต้นฉบับ
    for img_path in image_paths:
        try:
            # อ่านไฟล์รูปภาพ
            image = cv2.imread(img_path)
            # Albumentations ทำงานกับสีระบบ RGB แต่ OpenCV อ่านเป็น BGR จึงต้องแปลงสีก่อน
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # วนลูปเพื่อสร้างรูปภาพตามจำนวนที่กำหนด
            for i in range(IMAGES_TO_GENERATE_PER_SOURCE):
                # ใช้ pipeline ที่กำหนดไว้เพื่อสุ่มปรับแต่งรูปภาพ
                augmented_data = transform(image=image_rgb)
                augmented_image_rgb = augmented_data['image']
                
                # แปลงสีกลับเป็น BGR เพื่อให้ OpenCV บันทึกได้อย่างถูกต้อง
                augmented_image_bgr = cv2.cvtColor(augmented_image_rgb, cv2.COLOR_RGB2BGR)

                # สร้างชื่อไฟล์ใหม่ที่ไม่ซ้ำกัน
                base_name = os.path.basename(img_path)
                name, ext = os.path.splitext(base_name)
                output_filename = f"{name}_aug_{i+1}{ext}"
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)

                # บันทึกรูปภาพที่สร้างใหม่
                cv2.imwrite(output_path, augmented_image_bgr)
                total_generated += 1

            print(f"  - สร้าง {IMAGES_TO_GENERATE_PER_SOURCE} รูปจาก '{os.path.basename(img_path)}' เรียบร้อย")

        except Exception as e:
            print(f"เกิดข้อผิดพลาดกับไฟล์ '{os.path.basename(img_path)}': {e}")

    print("---------------------------------------------------------")
    print(f"🎉 สร้างรูปภาพทั้งหมด {total_generated} รูปเรียบร้อยแล้ว! ตรวจสอบได้ที่โฟลเดอร์ '{OUTPUT_FOLDER}'")