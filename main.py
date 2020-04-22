import math

import cv2.cv2 as cv2
import numpy as np
import pyautogui


def nothing(x):
    pass


cv2.namedWindow("tresholds")

cv2.createTrackbar("H_L", "tresholds", 0, 255, nothing)     #lower bound for hue
cv2.createTrackbar("H_U", "tresholds", 0, 255, nothing)     #upper bound for hue
cv2.createTrackbar("S_L", "tresholds", 0, 255, nothing)     #lower bound for saturation
cv2.createTrackbar("S_U", "tresholds", 0, 255, nothing)     #upper bound for saturation
cv2.createTrackbar("V_L", "tresholds", 0, 255, nothing)     #lower bound for value
cv2.createTrackbar("V_U", "tresholds", 0, 255, nothing)     #upper bound for value

cv2.setTrackbarPos("H_L", "tresholds", 0)
cv2.setTrackbarPos("H_U", "tresholds", 255)
cv2.setTrackbarPos("S_L", "tresholds", 0)
cv2.setTrackbarPos("S_U", "tresholds", 255)
cv2.setTrackbarPos("V_L", "tresholds", 174)
cv2.setTrackbarPos("V_U", "tresholds", 255)

cap = cv2.VideoCapture(0)

state1 = -1         # states of current hand gestures. State1 stores the state of previous frame. State2 stores the current state.
state2 = -1         # If state1 == state2, we can be sure that the correct gesture is detected. open hand=0, fist=1, thumb=2, #thumb&point=3
last_action = -1    # stores last executed action. move_cursor=0, click=1, double_click=2, right_click=3                                               
cursorX = 0     
cursorY = 0         #current cursor points
flag = 0            #flag that implies whether cursor commands are activated by pressing 's'

cx = 0
cy = 0              #points of the center of hand

rec_width = 450;
rec_heigth = 400;       # Setting a perfect threshold for all of the screen is impossible most of the time.
h, w, = (480, 640)      # So, you have to replace your hand inside the rectangular detection area at the left top corner.
                        # These variables are all about that rectange.
point1 = (0, 0)
point2 = (rec_width, rec_heigth)

heigth_ratio = 1080 / rec_heigth        # Since hand is detected in a limited rectangular area, these ratios provides full control
width_ratio = 1920 / rec_width          # to all of the screen 

radius = 100                # radius of an imaginary circle centered mass center of hand
ignore_arm_flag = 0         # a flag that implies whether ignore arm option is on/off


def rec_pos(event, x, y, flags, params):
    global point1, point2

    if event == cv2.EVENT_LBUTTONDBLCLK:

        p1 = x - int(rec_width / 2);
        p2 = y - int(rec_heigth / 2);
        p3 = x + int(rec_width / 2);
        p4 = y + int(rec_heigth / 2);

        if (p1 < 0):
            p1 = 0
            p3 = rec_width
        if (p2 < 0):
            p2 = 0
            p4 = rec_heigth
        if (p3 > w):
            p3 = w
            p1 = w - rec_width
        if (p4 > h):
            p4 = h
            p2 = h - rec_heigth

        point1 = (p1, p2)
        point2 = (p3, p4)


cv2.namedWindow("HSV_morph")
cv2.setMouseCallback("HSV_morph", rec_pos)

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)

    kernel = np.ones((3, 3), np.uint8)

    a, b = point1
    c, d = point2
    cropped = img[b:d, a:c]   #cropped part of detection area

    # ------------------------------------------------------------------------

    hl = cv2.getTrackbarPos("H_L", "tresholds")     #lower bound for hue
    hu = cv2.getTrackbarPos("H_U", "tresholds")     #upper bound for hue
    sl = cv2.getTrackbarPos("S_L", "tresholds")     #lower bound for saturation
    su = cv2.getTrackbarPos("S_U", "tresholds")     #upper bound for saturation
    vl = cv2.getTrackbarPos("V_L", "tresholds")     #lower bound for value
    vu = cv2.getTrackbarPos("V_U", "tresholds")     #upper bound for value

    lower = np.array([hl, sl, vl], dtype=np.uint8)
    upper = np.array([hu, su, vu], dtype=np.uint8)

    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)                      
    HSV_mask = cv2.inRange(img_HSV, lower, upper)
    HSV_blurred = cv2.GaussianBlur(HSV_mask, (5, 5), 100)  
    HSV_morph = cv2.morphologyEx(HSV_blurred, cv2.MORPH_OPEN, kernel)
    HSV_morph = cv2.dilate(HSV_morph, kernel)  

    cv2.rectangle(HSV_morph, point1, point2, (255, 0, 0), 2)

    # ------------------------------------------------------------------------

    hsv_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_cropped, lower, upper)
    mask_blurred = cv2.GaussianBlur(mask, (5, 5), 100)  
    mask_morph = cv2.morphologyEx(mask_blurred, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask_morph, kernel, iterations=1)  

    if ignore_arm_flag:
        mask[ignore_arm_point:rec_heigth, :] = 0
        HSV_morph[ignore_arm_point:480, :] = 0        #paint black ignored area

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_contour = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > max_area:
            max_area = area;
            max_contour = i;

    moments = cv2.moments(max_contour)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])           #Calculation for the center points of hand
        cy = int(moments['m01'] / moments['m00'])  

    center = (cx, cy)

    cv2.circle(cropped, center, 10, [0, 0, 255], 2)

    if max_area != 0:
        
        epsilon = 0.25 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)

        cv2.drawContours(cropped, [max_contour], -1, (0, 255, 0), 2)
        hull = cv2.convexHull(max_contour, returnPoints=True);
        hull_area = cv2.contourArea(hull)

        cv2.drawContours(cropped, [hull], 0, (0, 0, 255), 2)
        hull = cv2.convexHull(max_contour, returnPoints=False);
        defects = cv2.convexityDefects(max_contour, hull)


        if defects is not None:
            count = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]


                start = tuple(max_contour[s][0])
                end = tuple(max_contour[e][0])
                far = tuple(max_contour[f][0])

                triangle_area = (abs((start[0] * end[1] + end[0] * far[1] + far[0] * start[1]) - (    
                        end[0] * start[1] + far[0] * end[1] + start[0] * far[1]))) / 2     #calculation of triangular area consisting of start, end, far points

                if ((hull_area / triangle_area) < 30) & (far[1] <= cy) & (start[1]<=cy) & (end[1]<=cy):  # filteration of defect points
                    cv2.circle(cropped, far, 5, (255, 0, 0), -1)
                    count += 1

            font = cv2.FONT_HERSHEY_SIMPLEX

            if count >= 3:      # open hand detected

                state2 = 0

                last_action = 0

                if flag == 2:                               # if cursor mode is on
                    cursorX = int(cx * width_ratio)
                    cursorY = int(cy * heigth_ratio)
                    pyautogui.moveTo(cursorX, cursorY)

                elif flag == 1:                             # if terminal mode is on
                    cursorX = int(cx * width_ratio)
                    cursorY = int(cy * heigth_ratio)
                    print("move cursor to (", cursorX, ", ", cursorY, ")")

                if ignore_arm_flag:
                    radius = 100



            elif (count == 0):      # fist detected

                if (state1 == 1) & (last_action == 0):

                    last_action = 1
                    if flag == 2:
                        pyautogui.click(cursorX, cursorY)
                    elif flag == 1:
                        print("click on (", cursorX, ", ", cursorY, ")")

                state2 = 1

                if ignore_arm_flag:
                    radius = 70

            elif count == 1:  # thumb detected

                if (state1 == 2) & (last_action == 0):
                    last_action = 2
                    if flag == 2:
                        pyautogui.doubleClick(cursorX, cursorY)
                    elif flag == 1:
                        print("double click on (", cursorX, ", ", cursorY, ")")

                state2 = 2
                if ignore_arm_flag:
                    radius = 70

            elif count == 2:  # thumb&point finger detected

                if (state1 == 3) & (last_action == 0):
                    last_action = 3
                    if flag == 2:
                        pyautogui.rightClick(cursorX, cursorY)
                    elif flag == 1:
                        print("right click on (", cursorX, ", ", cursorY, ")")

                state2 = 3

                if ignore_arm_flag:
                    radius = 90

            if (state1 != -1) & (state1 == state2):
                if state1 == 0:
                    cv2.putText(img, "move", (30, 150), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                elif state1 == 1:
                    cv2.putText(img, "click", (30, 150), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                elif state1 == 2:
                    cv2.putText(img, "dbl click", (30, 150), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                else:
                    cv2.putText(img, "right click", (30, 150), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

            ignore_arm_point = cy + radius

            state1 = state2

    cv2.imshow("cropped", cropped)
    cv2.imshow("img", img)
    cv2.imshow("HSV_morph", HSV_morph)

    key = cv2.waitKey(1)

    if key == 115:                  # press 's' to switch between modes.
        if flag == 2:               # default mode : printing in the terminal
            flag = 0                # terminal mode : 
            print("flag = ", 0)     # cursor mode :
        elif flag == 0:
            flag = 1
            print("flag = ", 1)
        else:
            flag = 2
            print("flag = ", 2)

    elif key == 27:                 # press ESC to terminate the program
        break
    elif key == 116:                # press 't' to switch on/off the 'ignore arm feature'
        if ignore_arm_flag == 0:
            ignore_arm_flag = 1
            print("Trigger on!")
        else:
            ignore_arm_flag = 0
            radius = 100
            print("Trigger off!")

cap.release()
cv2.destroyAllWindows()
