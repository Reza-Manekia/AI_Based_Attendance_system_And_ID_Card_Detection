import cv2
import numpy as np
import tkinter as tk
from PIL import Image
import shutil
from tkinter import messagebox
from tkinter import simpledialog
import winsound
import csv
import os
import datetime

def real_time():
    realTimeDate = datetime.datetime.now()
    return realTimeDate

def faceDataGathering():
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height

    face_detector = cv2.CascadeClassifier('Cascades/Cascades/haarcascade_frontalface_default.xml')
    face_id = input('\n enter user id end press <return> ==> ')
    print("\n [INFO] Initializing face capture. Look the camera and wait ...")

    count = 0

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
        cv2.imshow('image', img)
        if cv2.waitKey(100) == 27:
            break
        elif count >= 30:
            print("\n [INFO] Exiting Program and cleanup stuff")
            break

    cam.release()
    cv2.destroyAllWindows()


def trainingModel():
    path = 'dataset'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("Cascades/Cascades/haarcascade_frontalface_default.xml");

    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)
                return faceSamples, ids

    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces, ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer/trainer.yml')
    print("Trained Succesfully")


def register_person():
    name = simpledialog.askstring("Registration", "Enter the name of the person:")
    Designation = simpledialog.askstring("Registration", "Enter the designation of the person:")
    id = int(simpledialog.askstring("Registration", "Enter the id of the person:"))

    if name and Designation and id:
        with open("authorized.txt", "a") as file:
            file.write(name + "\n")
            # messagebox.showinfo("Registration", f"{name} has been registered.")
        with open("Employees_data.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([name, Designation, id])

    faceDataGathering()
    trainingModel()

    # recognizer = cv2.face.LBPHFaceRecognizer_create()
    # recognizer.read('trainer/trainer.yml')
    # cascadePath = "Cascades/Cascades/haarcascade_frontalface_default.xml"
    # faceCascade = cv2.CascadeClassifier(cascadePath);
    names = ['None']
    names.insert(int(id), name)
    messagebox.showinfo("Registration", f"{name} has been registered.")


def Attendance():

    with open("authorized.txt", "r") as file:
        authorized_persons = file.read().splitlines()


    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        date_time = real_time()
        ret, frame = cap.read()
        qr = cv2.QRCodeDetector()
        v1, v2, v3 = qr.detectAndDecode(frame)
        if v1 != "":
            if v1 in authorized_persons:
                messagebox.showinfo("Authorization", f"Authorized: {v1}")
                original_file = "Employees_data.csv"
                temp_file = "temp.csv"

                with open(original_file, "r") as file:
                    reader = csv.reader(file)
                    rows = list(reader)
                    for row in rows:
                        if len(row) > 0 and row[0].strip() == v1.strip():
                            row.append(date_time)
                            row.append("p")
                            break

                with open(temp_file, "w", newline='') as temp:
                    writer = csv.writer(temp)
                    writer.writerows(rows)



                # Replace the original file with the temporary file
                shutil.move(temp_file, original_file)

                messagebox.showinfo("Authorization", f"{v1} Qr Code detected.")
                winsound.Beep(1000, 500)
                break
            else:
                messagebox.showinfo("Authorization", "Unauthorized")
                break
        cv2.imshow('Barcode Scanning', frame)

        if cv2.waitKey(1) == 27:
            break

    cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "Cascades/Cascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height

    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            if (100 - confidence > 35):
                messagebox.showinfo("Attendance", f"{v1} marked as present.")
                winsound.Beep(1000, 500)
                break

        cv2.imshow('camera', img)
        if cv2.waitKey(1) == 27:
            break
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()




window = tk.Tk()
window.title("Barcode Scanning")

# Set window size and position
window.geometry("550x350")
window.resizable(False, False)

# Create a heading label
heading_label = tk.Label(window, text="ADVANCE AI BASED ATTENDANCE SYSTEM", font=("Arial", 15, "bold"), pady=20)
heading_label.pack()

# Create a button to start barcode detection
start_button = tk.Button(window, text="Start Detection", font=("Arial", 12), command=Attendance)
start_button.pack(pady=20)

register_button = tk.Button(window, text="Registration", font=("Arial", 12), command=register_person)
register_button.pack(pady=10)

window.mainloop()