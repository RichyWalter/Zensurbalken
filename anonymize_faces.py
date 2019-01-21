# cli based tool to anonymize faces

# imports
from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np


# init the face detection module
detector = MTCNN()

# TODO hardcoded values, fix that with relative values
def draw_pornobalken(l_eye, r_eye, img):
    l_eye = (l_eye[0]- 10, l_eye[1] + 10)
    r_eye = (r_eye[0]+ 10, r_eye[1] - 10)
    cv2.rectangle(img,l_eye,r_eye,(0,0,0),-1)
    return img

#face = (x,y,width,height)
def pixelate_face(img, face):
    heigth = face[3]
    width = face[2]
    print(face)
    pixel_size = int(max(heigth, width)/10)
    for i in range(face[1], face[1]+heigth, pixel_size):
        for y in range(pixel_size):
            img[i+y] = img[i]
            print(img[i][10])
    return img




# detect and anonymize faces on an image/frame
def detect_and_anonymize(img):
    # detect faces on the image/frame
    faces = detector.detect_faces(img)
    ano_img = img
    # draw black rectangles on each face
    for face in faces:
        left_eye = face["keypoints"]["left_eye"]
        right_eye = face["keypoints"]["right_eye"]
        bnd_box = face["box"]
        #ano_img = draw_pornobalken(left_eye, right_eye, ano_img)
        ano_img = pixelate_face(ano_img, bnd_box)
        
    return ano_img 


MODE = "image"  # webcam, image_folder, image

if(MODE == "image"):
    single_img = cv2.imread("image_data/testbild.jpg")
    result = detect_and_anonymize(single_img)
    # show result
    cv2.imshow('result',result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif(MODE == "webcam"):
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # anonymize frame from webcam stream
        anonym_stream = detect_and_anonymize(frame)

        # Display the resulting frame
        cv2.imshow('frame',anonym_stream)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
