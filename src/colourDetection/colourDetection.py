import cv2
import pandas as pd

import getColour

def draw_function(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b, g, r, xpos, ypos, clicked

        clicked = True
        xpos = x
        ypos = y

        b,g,r = img[y,x]
        b = int(b)
        g = int(g)
        r = int(r)

img = cv2.imread('../../images/colours.jpg')

clicked = False
r = g = b = xpos = ypos = 0

index=["colour", "colour_name", "hex", "R", "G", "B"]
csv = pd.read_csv('../../data/colours.csv', names=index, header=None)
       
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_function)

while(True):
    cv2.imshow("image", img)
    if (clicked):
        cv2.rectangle(img, (20,20), (750,60), (b,g,r), -1)
        text = getColour.getColourName(r, g, b, csv) + ' R=' + str(r) +  ' G=' + str(g) +  ' B=' + str(b)
        cv2.putText(img, text, (50, 50), 2, 0.8, (255,255,255), 2, cv2.LINE_AA)

        if(r+g+b >= 600):
            cv2.putText(img, text, (50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                
        clicked=False
    
    if cv2.waitKey(20) & 0xFF == 27:
        break
    
cv2.destroyAllWindows()
