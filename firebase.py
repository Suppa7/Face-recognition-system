import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("service_firebase.json")
firebase_admin.initialize_app(cred,{
    'databaseURL' : "https://face-recognition-project-d959a-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

ref = db.reference('Students')

data = {
    "0001" : {
        "ID" : "6510510241",
        "Name" : "Suppakorn krobpech",
        "role" : "Student",
        "major" : "Bachelor of Business Administration ",
        "year" : "4",
        "total_attendance" : 1,
        "last_attendance_time" : "2025-08-12 12:00:00"
    },
    "0002" : {
        "ID" : "6510510212",
        "Name" : "Yodsawit Khaothong",
        "role" : "Student",
        "major" : "Bachelor of Business Administration ",
        "year" : "4",
        "total_attendance" : 1,
        "last_attendance_time" : "2025-08-12 12:00:00"
    },
    "0003" : {
        "ID" : "10510134",
        "Name" : "Dr. Kannikar Paripremkul",
        "role" : "Professor",
        "major" : "Bachelor of Business Administration ",
        "year" : "no year",
        "total_attendance" : 1,
        "last_attendance_time" : "2025-08-12 12:00:00"
    }
}

for key, value in data.items():
    ref.child(key).set(value)