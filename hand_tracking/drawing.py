import os
import uuid

import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

hands_array =  np.zeros((2,21,2))
shape_records = [[],[],[]]
shape=""
record=False

def get_label(index, results):
    output = None

    try:
        output= results.multi_handedness[index].classification[0].label
    except:
         pass
    return output

def get_point_distance(point1, point2):
    
    return math.hypot(point1[0]-point2[0], point1[1]-point2[1])

def get_finger_angles(joint_list):
    output = []
    for joint in joint_list:
        a = np.array((joint[0][0], joint[0][1]))
        b = np.array((joint[1][0], joint[1][1]))
        c = np.array((joint[2][0], joint[2][1]))

        rad =  np.arctan2(c[1]- b[1], c[0]- c[0]) - np.arctan2(a[1]- b[1], a[0]-b[0])
        angle = np.abs(rad*180/np.pi)

        if angle>180:
            angle = 360- angle 
        output.append(angle)
    return output

def draw_records(plain_image):
    image = plain_image
    if len(shape_records[0])>0:
        for measurements in shape_records[0]:
            image = cv2.line(image, measurements[0], measurements[1], (255,255,255), 2, cv2.LINE_4)
    if len(shape_records[1])>0:
        for measurements in shape_records[1]:
            image = cv2.circle(image, measurements[0], measurements[1], (255,255,255), 2, cv2.LINE_4)

    if len(shape_records[2])>0:
        for measurements in shape_records[2]:
            image = cv2.rectangle(image, measurements[0], measurements[1], (255,255,255), 2, cv2.LINE_4)
    
    return image
        

with mp_hands.Hands(min_detection_confidence= 0.8, min_tracking_confidence= 0.5) as hands:
    while cap.isOpened():
        hands_array = np.zeros((2,21,2))
        shape= ""
        
        res, frame = cap.read()

        image = cv2.cvtColor(frame,  cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)

        image.flags.writeable = False

        results = hands.process(image)
        
        image = cv2.cvtColor(image,  cv2.COLOR_RGB2BGR)
        image.flags.writeable = True

        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2))
                
                #calculates landmarks array
            
                label =  get_label(num, results)
                if label:
                    if label == "Left":
                        hand_idx= 0
                    elif label == "Right":
                        hand_idx = 1
                    for i,landmark in enumerate(hand.landmark):
                        landmark= np.multiply(np.array((landmark.x, landmark.y)),
                                               np.array((cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))).astype(int)
                        hands_array[hand_idx,i] = landmark
  
            if(hands_array[0,0,0] != 0) and (hands_array[1,0,0] != 0):

                #draws line
                if(get_point_distance(hands_array[0,4],hands_array[0,8]) <= 30) and (get_point_distance(hands_array[1,4],hands_array[1,8]) <= 30):
                    cv2.line(image, tuple((int(hands_array[0,8,0]), int(hands_array[0,8,1]))), tuple((int(hands_array[1,8,0]), int(hands_array[1,8,1]))), (255,255,255), 2, cv2.LINE_4)
                    shape = "line"
                #draws circle
                elif(get_point_distance(hands_array[0,4],hands_array[0,6]) <= 35) and (get_point_distance(hands_array[1,4],hands_array[1,8]) <= 30):
                    cv2.circle(image, tuple((int(hands_array[0,8,0]), int(hands_array[0,8,1]))), int(get_point_distance(hands_array[0,8], hands_array[1,4])), (255,255,255), 2, cv2.LINE_4 )
                    shape = "circle1"
                elif(get_point_distance(hands_array[0,4],hands_array[0,8]) <= 30) and (get_point_distance(hands_array[1,4],hands_array[1,6]) <= 35):
                    cv2.circle(image, tuple((int(hands_array[1,8,0]), int(hands_array[1,8,1]))), int(get_point_distance(hands_array[0,4], hands_array[1,8])), (255,255,255), 2, cv2.LINE_4 )
                    shape= "circle2"

                #draws rectangle
                joint_list = ((hands_array[0,3],hands_array[0,2], hands_array[0,6]), (hands_array[1,3],hands_array[1,2], hands_array[1,6]))
                angles = get_finger_angles(joint_list)
                left_angle= angles[0]
                right_angle = angles[1]

                if (left_angle > 60 and right_angle > 60 ) and ((math.fabs(hands_array[0,13,1] - hands_array[0,0,1])< 30) and (math.fabs(hands_array[1,13,1] - hands_array[1,0,1])< 30)):
                    cv2.rectangle(image, tuple((int(hands_array[0,2,0]), int(hands_array[0,2,1]))), tuple((int(hands_array[1,2,0]), int(hands_array[1,2,1]))), (255,255,255), 2, cv2.LINE_4)
                    shape= "rectangle"
                #recording drawings
                joint_list= ((hands_array[0,17], hands_array[0,18], hands_array[0,19]), (hands_array[1,17], hands_array[1,18], hands_array[1,19]))
                angles = get_finger_angles(joint_list)
                left_angle = angles[0]
                right_angle = angles[1]
                if(left_angle < 110 and right_angle < 110) and record == False:
                    record= True
                    if shape == "line":
                        shape_records[0].append(tuple((tuple((int(hands_array[0,8,0]), int(hands_array[0,8,1]))), tuple((int(hands_array[1,8,0]), int(hands_array[1,8,1]))))))
                    elif shape == "circle1":
                        shape_records[1].append(tuple((tuple((int(hands_array[0,8,0]), int(hands_array[0,8,1]))), int(get_point_distance(hands_array[0,8], hands_array[1,4])))))
                    elif shape == "circle2":
                        shape_records[1].append(tuple((tuple((int(hands_array[1,8,0]), int(hands_array[1,8,1]))), int(get_point_distance(hands_array[0,4], hands_array[1,8])))))
                    elif shape == "rectangle":
                        shape_records[2].append(tuple(( tuple((int(hands_array[0,2,0]), int(hands_array[0,2,1]))), tuple((int(hands_array[1,2,0]), int(hands_array[1,2,1]))))))
                    else:
                        pass

                elif  (left_angle >110 and right_angle > 110):
                    record = False

                joint_list= ((hands_array[0,9], hands_array[0,10], hands_array[0,11]), (hands_array[1,9], hands_array[1,10], hands_array[1,11]))
                angles = get_finger_angles(joint_list)
                left_angle = angles[0]
                right_angle = angles[1]
                if(left_angle < 110 and right_angle < 110):
                    shape_records = [[],[],[]]

                

        image = draw_records(image)

        cv2.imshow("Hand_Tracking", image)

        nah = (((hands_array[1,4,0]- hands_array[1, 10, 0])< 0 and (hands_array[1,4,0]- hands_array[1, 10, 0])> -30)  and (hands_array [1,4,0]-hands_array[1,6,0])<30) and math.fabs(hands_array[1,4,1]- hands_array[1,10,1])<30
        if(cv2.waitKey(10) & 0xFF == ord('q')):
                break
        
cap.release()
cv2.destroyAllWindows()

