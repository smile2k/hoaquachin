"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import cv2 as cv
import numpy as np


def main(argv):
    default_file = 'E:\\20211\\xulyanh\\image\\detect-simple-shapes-src-img.png'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src1 = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
    src = cv.GaussianBlur(src1, (7, 7), 1)
    src = cv.cvtColor(src1, cv.COLOR_BGR2GRAY)
    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        print('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1
    src = cv.GaussianBlur(src,(7,7),1)
    dst = cv.Canny(src, 50, 50, 3)
    #contours, _ = cv.findContours(src, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours, _ = cv.findContours(dst, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 100:
            cv.drawContours(src1, cnt, -1, (0, 255, 0), 2)  # vẽ lại ảnh contour vào ảnh gốc
            peri = cv.arcLength(cnt,True)
            approx = cv.approxPolyDP(cnt, 0.02*peri,True)
            obj = len(approx)
            print(obj)
            x,y,w,h = cv.boundingRect(approx)
            if obj == 3:
                objectType = "Tri"
            elif obj == 4:
                dieukien = w/(float(h))
                if 0.98 < dieukien < 1.05:
                    objectType = "square"
                else:
                    objectType = "rectangle"
            elif obj == 5:
                objectType = "penta"
            elif obj == 6:
                objectType = "hexa"
            else:
                objectType = "circle"

            cv.putText(src1, objectType, (x+(w//2)-10,y+(h//2)),cv.FONT_HERSHEY_COMPLEX, 0.5,(0,255,0),1)
    cv.imshow("ve contours", src1)
    cv.waitKey()
    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 50, 10)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, 50, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(src1, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 3, cv.LINE_AA)
            #print(i)

    #cv.imshow("Source", src)
    #cv.imshow("Standard Hough Line Transform", cdst)
    #cv.imshow("Probabilistic Line Transform", src1)

    cv.waitKey()
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
