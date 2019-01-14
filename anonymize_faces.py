#cli based tool to anonymize faces

#imports
from mtcnn.mtcnn import MTCNN
import cv2

#read test image 
img = cv2.imread("image_data/testbild.jpg")

#init the face detection module
detector = MTCNN()
#detect faces on the image/frame
faces = detector.detect_faces(img)

#TODO hardcoded values, fix that with relative values
def draw_pornobalken(l_eye, r_eye):
    l_eye = (l_eye[0]- 10, l_eye[1] + 10)
    r_eye = (r_eye[0]+ 10, r_eye[1] - 10)
    cv2.rectangle(img,l_eye,r_eye,(0,0,0),-1)

#draw black rectangles on each face
for face in faces:
    left_eye = face["keypoints"]["left_eye"]
    right_eye = face["keypoints"]["right_eye"]
    draw_pornobalken(left_eye, right_eye)

#show result
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()