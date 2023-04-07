import cv2
import cvzone
from cvzone import FaceDetectionModule as fp
from cvzone import FaceMeshModule as fm

cap = cv2.VideoCapture(0)

detector = fp.FaceDetector()
hellow = fm.FaceMeshDetector()
while True:
    succ , img = cap.read()
    sac = img.copy()
    img1, bbox1 = detector.findFaces(img)
    img2 , bbox2 = hellow.findFaceMesh(sac)
    cv2.imshow("face", img1)
    cv2.imshow("facemesh", img2)
    cv2.waitKey(1)