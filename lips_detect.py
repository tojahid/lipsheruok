"""
    ReadMe==>
        1.Lips detect
        2. Color based on hex code
        3. realtime
"""

import cv2
import numpy as np
import dlib
from PIL import ImageColor
from datetime import datetime

webcam = True
onlyLipsColor = True
Lips_Color = "9d65c9"
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./static/shape_predictor_68_face_landmarks.dat")



def Hex2rgb(hex):

    if hex.startswith("#"):
        rgb = ImageColor.getcolor(hex, "RGB")
    else:
        hex = "#"+hex
        rgb = ImageColor.getcolor(hex, "RGB")

    r, g, b = rgb[0], rgb[1], rgb[2]
    return rgb, r, g, b

def createBox(img,points,scale=5,masked= False,cropped= True):
    if masked:
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask,[points],(255,255,255))
        img = cv2.bitwise_and(img,mask)
        #cv2.imshow('Mask',mask)

    if cropped:
        bbox = cv2.boundingRect(points)
        x, y, w, h = bbox
        imgCrop = img[y:y+h,x:x+w]
        imgCrop = cv2.resize(imgCrop,(0,0),None,scale,scale)
        cv2.imwrite("Mask.jpg",imgCrop)
        return imgCrop
    else:
        return mask

def lips_changer(image_path, Lips_ColorHex):

    img = cv2.imread(image_path)
    print(img)
    img = cv2.resize(img,(0,0),None,0.8,0.8)
    imgOriginal = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(imgOriginal)
    for face in faces:
        x1,y1 = face.left(),face.top()
        x2,y2 = face.right(),face.bottom()
        #imgOriginal=cv2.rectangle(imgOriginal, (x1, y1), (x2, y2), (0, 255, 0), 2)
        landmarks = predictor(imgGray, face)

        myPoints =[]
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            myPoints.append([x,y])
            

        if len(myPoints) != 0:
                try:
                    myPoints = np.array(myPoints)
                    imgLips = createBox(img, myPoints[48:61], 3, masked=True, cropped=False)

                    imgColorLips = np.zeros_like(imgLips)
                    rgb, r, g, b = Hex2rgb(hex = Lips_ColorHex)  # color converter
                    imgColorLips[:] = b, g, r
                    imgColorLips = cv2.bitwise_and(imgLips, imgColorLips)
                    imgColorLips = cv2.GaussianBlur(imgColorLips, (7, 7), 10)


                    if onlyLipsColor:
                        imgOriginalGray = cv2.cvtColor(imgOriginal,cv2.COLOR_BGR2GRAY)
                        imgOriginalGray = cv2.cvtColor(imgOriginalGray, cv2.COLOR_GRAY2BGR)
                        imgColorLips = cv2.addWeighted(imgOriginalGray ,1,imgColorLips,0.4,0)
                    else:
                        imgColorLips = cv2.addWeighted(imgOriginal, 1, imgColorLips, 0.4, 0)

                    return imgColorLips

                    # #cv2.imshow('Lips', imgLips)

                except:
                    pass


def save_image(image_path, hexcode):
    print("PATH OF IMATE IN SAVEIMAGE"+image_path);
    print(hexcode)
    img = lips_changer(image_path, hexcode)
    indx = np.random.randint(10000)
    # print(datetime.now().strftime('%h'));
    path = "./static/LipsImage/"+ datetime.now().strftime('%m%S%M')+".jpg"
    cv2.imwrite(path,img)


if __name__ == '__main__':
    save_image()

cap.release()
cv2.destroyAllWindows()
