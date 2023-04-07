import cv2                                            #import opencv module cv2 
import cvzone                                         #importing cvzone
from cvzone import FaceDetectionModule as fp
from cvzone import FaceMeshModule as fm

cap = cv2.VideoCapture(0)                              #getting the webcam feed by setting 0  for laptop default webcam and by 1 the external webcam

detector = fp.FaceDetector()
hellow = fm.FaceMeshDetector()                          #initializing facedetection 
while True:
    succ , img = cap.read()
    sac = img.copy()
    img1, bbox1 = detector.findFaces(img)               #facedetection
    img2 , bbox2 = hellow.findFaceMesh(sac)             #facemeshdetection
    cv2.imshow("face", img1)
    cv2.imshow("facemesh", img2)
    cv2.waitKey(1)
