import cv2 
import numpy as np
import mediapipe as mp
import time
import math

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
load_dotenv()

def get_label(hand_idx, results):
    
    label = results.multi_handedness[hand_idx].classification[0].label
    if label == "Left":
        hand_num = 0
    elif label == "Right":
        hand_num = 1

    return hand_num, label

def get_up_fingers(hands_array, hand_num):
    fingertips= (4,8,12,16,20)
    open_fingers= []

    if (hand_num == 0 and hands_array[hand_num, 4, 0] > hands_array[hand_num, 1, 0]) or (hand_num == 1 and hands_array[hand_num, 4, 0] < hands_array[hand_num, 1, 0]):
        open_fingers.append(0)

    for i in range(1,5):
        if hands_array[hand_num, fingertips[i], 1] < hands_array[hand_num, fingertips[i]-3, 1]:
            open_fingers.append(i)
    return open_fingers


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

scope= "user-modify-playback-state streaming"
auth_manager = SpotifyOAuth(scope=scope)
sp = spotipy.Spotify(auth_manager=auth_manager)

cap = cv2.VideoCapture(0)
hands_array =  np.zeros((2,21,2))
fingers_up = [[],[]]
playing = True
lastRequesTime = time.perf_counter()
lastRequest = ""
volume = 50
last_volume = 50


with mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5) as hands:
    while cap.isOpened():
        hands_array = np.zeros((2,21,2))
        fingers_up = [[],[]]

        res, image =  cap.read()
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        image.flags.writeable = False

        results =  hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_idx, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                          mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2))
        
                hand_num, label = get_label(hand_idx, results)
                for i, landmark in enumerate(hand.landmark):
                    landmark = np.multiply(np.array((landmark.x, landmark.y)),
                                           np.array((cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))).astype(int)
                    hands_array[hand_num, i] = landmark

                fingers_up[hand_num] = get_up_fingers(hands_array, hand_num)

        else:
            hands_array = np.zeros((2,21,2))

        if (hands_array[0,0,0] != 0) or (hands_array[1,0,0] != 0):
            if ((playing and (not fingers_up[1])) and time.perf_counter() - lastRequesTime > 1) and lastRequest !="stop":
                lastRequest = "stop"
                playing= False
                lastRequesTime = time.perf_counter()
                sp.pause_playback()

            elif (((not playing) and fingers_up[1] == [0]) and time.perf_counter() - lastRequesTime > 1) and lastRequest !="continue":
                lastRequest = "continue"
                playing=True
                lastRequesTime = time.perf_counter()
                sp.start_playback()

            elif ((fingers_up[0] == [1,2,3,4] and time.perf_counter()- lastRequesTime > 1) and hands_array[1,0,0] == 0) and lastRequest !="last":
                lastRequest = "last"
                lastRequesTime = time.perf_counter()
                sp.previous_track()

            elif ((fingers_up[1] == [1,2,3,4] and time.perf_counter()- lastRequesTime > 1) and hands_array[0,0,0] == 0) and lastRequest !="next":
                lastRequest = "next"
                lastRequesTime = time.perf_counter()
                sp.next_track()

            if math.hypot(math.fabs(hands_array[1,4,0] - hands_array[1,8,0]), math.fabs(hands_array[1,4,1] - hands_array[1,8,1])) <= 15:
                if hands_array[1,4,1] < 100:
                    volume= 100
                elif hands_array[1,4,1] > 400:
                    volume = 0
                else:
                    volume = round((400- hands_array[1,4,1]) / 3)

                if ((volume//10 != last_volume //10 )and math.fabs(volume - last_volume ) > 5) and time.perf_counter() - lastRequesTime > 1:
                    last_volume = volume
                    last_request_time = time.perf_counter();
                    sp.volume(volume)

        cv2.imshow("Hand_Tracking", image)
        print(volume)

        #nah = (((hands_array[1,4,0]- hands_array[1, 10, 0])< 0 and (hands_array[1,4,0]- hands_array[1, 10, 0])> -30)  and (hands_array [1,4,0]-hands_array[1,6,0])<30) and math.fabs(hands_array[1,4,1]- hands_array[1,10,1])<30
        if(cv2.waitKey(10) & 0xFF == ord('q')):
                break