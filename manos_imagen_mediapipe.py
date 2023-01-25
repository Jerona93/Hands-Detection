import cv2              #   Import OpenCv, it installs only with mediapipe
import mediapipe as mp  #   Import mediapipe and give it an alias

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands(
    static_image_mode = True,
    max_num_hands = 2,      #    Hands to be recognized
    min_detection_confidence = 0.5) as hands:
    # try with two pics, with hands and without hand, to try hands detection
    image = cv2.imread('image1.jpg')

    height, width, _ = image.shape
    image = cv2.flip(image,1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    #   Handedness, information about which hand is
    print('Handedness', results.multi_handedness)

    #   Hand LandMarks, list of the 21 points of the hand
    # print('Hand LandMarks:', results.multi_hand_landmarks)
    #   Condition in error case:
    if results.multi_hand_landmarks is not None:
        index = [4,8,12,16,20] # Path 2
        for hand_landmarks in results.multi_hand_landmarks:
        #------To draw the points and connections------
            # #print(hand_landmarks)
            # mp_drawing.draw_landmarks(
            #     image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            #     #To change connections colors
            #     mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=4, circle_radius=2),
            #     mp_drawing.DrawingSpec(color=(255, 49, 12), thickness=4)

            #     )
        #------ To access the points by their names (names in documentation)------
            # x1 =int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * width) # Obtain coordinates and convert to integers
            # y1 =int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * height) 

            # x2 =int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
            # y2 =int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)

            # x3 =int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * width)
            # y3 =int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * height)

            # x4 =int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * width)
            # y4 =int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * height)

            # x5 =int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * width)
            # y5 =int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * height)

            # print(x1, y1)
            # Draw circle in a coordinate
            # cv2.circle(image, (x1,y1), 3, (255,0,0,0), 3)
            # cv2.circle(image, (x2,y2), 3, (255,0,0,0), 3)
            # cv2.circle(image, (x3,y3), 3, (255,0,0,0), 3)
            # cv2.circle(image, (x4,y4), 3, (255,0,0,0), 3)
            # cv2.circle(image, (x5,y5), 3, (255,0,0,0), 3)
        # More simply (path 2)
            for (i, points) in enumerate(hand_landmarks.landmark):
                if i in index:
                    x = int(points.x * width)
                    y = int(points.y * height)
                    cv2.circle(image, (x,y), 3, (255,0,0,0), 3)
            

    image = cv2.flip(image,1)
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
