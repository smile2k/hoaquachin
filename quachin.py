from cv2 import cv2
import numpy as np

img = cv2.imread("E:\\20211\\xulyanh\\image\\qua3.PNG")
hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

low_red = np.array([0,150,20]) #161,55,84 red , quả chín 0,150,20
high_red = np.array([15,255,255]) #179,255,255 red, quả chín 15,255,255
red_mask = cv2.inRange(hsv_frame,low_red,high_red)
contours1, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(contours1) != 0:
    for contour in contours1:
        if cv2.contourArea(contour) > 1200:
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,0,255),3)
            cv2.putText(img, "ripen ", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
red = cv2.bitwise_and(img, img, mask=red_mask)

cv2.imshow("Frame", img)
cv2.waitKey()
