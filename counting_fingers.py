import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees

#------ calculate the center
def palm_centroid(coordinates_list):
    coordenates = np.array(coordinates_list)
    centroid = np.mean(coordenates,axis=0)
    centroid = int(centroid[0]), int(centroid[1])
    return centroid

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

#------ Figers ------
thumb_points = [1, 2, 4,]
palm_points = [0, 1, 2, 5, 9, 13, 17]
fingertips_points = [8, 12, 16, 20]
finger_base_points = [6, 10, 14,18]

#------ Colors ------
GREEN = (48, 255, 48)
BLUE = (192, 101, 21)
YELLOW = (8, 284, 255)
PURPLE = (128, 64, 128)
PEACH = (180, 229, 255)

with mp_hands.Hands(
    model_complexity = 1,
    max_num_hands = 1,      #    Hands number
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5) as hands:

    while True:
        ret , frame = cap.read()
        if ret == False:
            break
        
        height, width, _ = frame.shape
        frame = cv2.flip(frame,1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        fingers_counter = '_'
        thickness = [2, 2, 2, 2, 2]

        if results.multi_hand_landmarks:
            coordenates_thumb = []
            coordenates_palm = []
            coordenates_ft = []
            coordenates_fb = []
            
            for hand_landmarks in results.multi_hand_landmarks:
                #----- Coordinates of each finger
                for index in thumb_points:
                    x =int(hand_landmarks.landmark[index].x * width)
                    y =int(hand_landmarks.landmark[index].y * height)
                    coordenates_thumb.append([x, y])

                for index in palm_points:
                    x =int(hand_landmarks.landmark[index].x * width)
                    y =int(hand_landmarks.landmark[index].y * height)
                    coordenates_palm.append([x, y])

                for index in fingertips_points:
                    x =int(hand_landmarks.landmark[index].x * width)
                    y =int(hand_landmarks.landmark[index].y * height)
                    coordenates_ft.append([x, y])

                for index in finger_base_points:
                    x =int(hand_landmarks.landmark[index].x * width)
                    y =int(hand_landmarks.landmark[index].y * height)
                    coordenates_fb.append([x, y])
                #---------
                #THUMB
                p1 = np.array(coordenates_thumb[0])
                p2 = np.array(coordenates_thumb[1])
                p3 = np.array(coordenates_thumb[2])

                l1 = np.linalg.norm(p2 - p3)
                l2 = np.linalg.norm(p1 - p3)
                l3 = np.linalg.norm(p1 - p2)
                #---- Angle calculate
                angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                #print(angle)
                thumb_finger = np.array(False)
                if angle > 150:
                    thumb_finger = np.array(True)
                #print('thumb_finger:', thumb_finger)
                #-------
                nx, ny = palm_centroid(coordenates_palm)
                cv2.circle(frame, (nx, ny), 3, (255,0,0,0), 3)
                coordenates_centroid = np.array([nx, ny])
                coordenates_ft = np.array(coordenates_ft)
                coordenates_fb = np.array(coordenates_fb)

                #------ Distance calculate
                d_centrid_ft = np.linalg.norm(coordenates_centroid - coordenates_ft, axis=1)
                d_centrid_fb = np.linalg.norm(coordenates_centroid - coordenates_fb, axis=1)
                dif = d_centrid_ft - d_centrid_fb
                fingers = dif > 0
                fingers = np.append(thumb_finger, fingers) #    add thumb finger
                #print('dif: ',fingers)
                fingers_counter = str(np.count_nonzero(fingers == True))
                for (i, finger) in enumerate(fingers):
                    if finger == True:
                        thickness[i] = -1
                #------
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                    # mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=4, circle_radius=2),
                    # mp_drawing.DrawingSpec(color=(255, 49, 12), thickness=4))
                
        #-----Visualization
        cv2.rectangle(frame, (0, 0), (88, 88), (125,220,0), -1)
        cv2.putText(frame, fingers_counter,(15, 65), 1, 5, (255, 255, 255), 2)
        # Thumb
        cv2.rectangle(frame, (100, 10), (150, 60), PEACH, thickness[0])
        cv2.putText(frame, "Thumb",(100, 80), 1, 1, (255, 255, 255), 2)
        # Index
        cv2.rectangle(frame, (160, 10), (210, 60), PURPLE, thickness[1])
        cv2.putText(frame, "Index",(160, 80), 1, 1, (255, 255, 255), 2)
        # Heart
        cv2.rectangle(frame, (220, 10), (270, 60), YELLOW, thickness[2])
        cv2.putText(frame, "Heart",(220, 80), 1, 1, (255, 255, 255), 2)
        # Annular
        cv2.rectangle(frame, (280, 10), (330, 60), GREEN, thickness[3])
        cv2.putText(frame, "Annular",(280, 80), 1, 1, (255, 255, 255), 2)
        # Little
        cv2.rectangle(frame, (340, 10), (390, 60), BLUE, thickness[4])
        cv2.putText(frame, "Little",(350, 80), 1, 1, (255, 255, 255), 2)
        #------
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
cap.release()
cv2.destroyAllWindows()
