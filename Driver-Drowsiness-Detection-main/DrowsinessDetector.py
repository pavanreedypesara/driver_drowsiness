from tkinter import *
import tkinter
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import winsound  # Replacing playsound with winsound (Windows only)

# Initialize GUI
main = tkinter.Tk()
main.title("Driver Drowsiness Monitoring")
main.geometry("500x400")

# Function to calculate Eye Aspect Ratio (EAR)
def EAR(eye):
    point1 = dist.euclidean(eye[1], eye[5])
    point2 = dist.euclidean(eye[2], eye[4])
    distance = dist.euclidean(eye[0], eye[3])
    ear = (point1 + point2) / (2.0 * distance)
    return ear

# Function to calculate Mouth Aspect Ratio (MAR)
def MAR(mouth):
    point   = dist.euclidean(mouth[0], mouth[6])  # Horizontal distance
    point1  = dist.euclidean(mouth[2], mouth[10]) # Vertical distance 1
    point2  = dist.euclidean(mouth[4], mouth[8])  # Vertical distance 2
    mar = (point1 + point2) / (2.0 * point)  # Mouth aspect ratio
    return mar

# Function to play alarm sound
def play_alarm():
    winsound.PlaySound("alarm.wav", winsound.SND_ASYNC)

# Function to start monitoring
def startMonitoring():
    pathlabel.config(text="Webcam Connected Successfully")
    webcam = cv2.VideoCapture(0)
    svm_predictor_path = 'SVMclassifier.dat'

    # Drowsiness detection parameters
    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 10
    MOU_AR_THRESH = 0.50  # Adjusted for better yawning detection

    COUNTER = 0
    yawn_status = False
    yawns = 0

    svm_detector = dlib.get_frontal_face_detector()
    svm_predictor = dlib.shape_predictor(svm_predictor_path)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        prev_yawn_status = yawn_status
        rects = svm_detector(gray, 0)

        for rect in rects:
            shape = svm_predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]

            leftEAR = EAR(leftEye)
            rightEAR = EAR(rightEye)
            mouthMAR = MAR(mouth)

            ear = (leftEAR + rightEAR) / 2.0

            # Draw eye contours
            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 255), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 255), 1)
            cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 0), 1)

            # Eye Drowsiness Detection
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                cv2.putText(frame, "Eyes Closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    play_alarm()
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display EAR value
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Yawning Detection
            if mouthMAR > MOU_AR_THRESH:
                play_alarm()
                cv2.putText(frame, "Yawning, DROWSINESS ALERT!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                yawn_status = True
            else:
                yawn_status = False

            if prev_yawn_status and not yawn_status:
                yawns += 1

            # Display MAR value
            cv2.putText(frame, "MAR: {:.2f}".format(mouthMAR), (480, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Yawn Count: {}".format(yawns), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    webcam.release()

# GUI Elements
font = ('times', 16, 'bold')
title = Label(main, text='Driver Drowsiness Monitoring System using Visual Behaviour and Machine Learning',
              bg='black', fg='white', font=font, height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Start Behaviour Monitoring Using Webcam", command=startMonitoring, font=font1)
upload.place(x=50, y=200)

pathlabel = Label(main, bg='DarkOrange1', fg='white', font=font1)
pathlabel.place(x=50, y=250)

main.config(bg='chocolate1')
main.mainloop()
