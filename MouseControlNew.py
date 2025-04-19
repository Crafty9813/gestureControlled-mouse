import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
index_y_prev, index_x_prev = 0, 0
smoothing_factor = 0.5
click_threshold = 30
click_delay = 1
movement_threshold = 70 #px

landmark_drawing_spec = drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)

while True:

    ret, frame = cap.read()
    #frame.flags.writeable = False
    frame = cv2.flip(frame, 1) # otherwise left and right movement would be opposite
    frame_height, frame_width, ret = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:

            drawing_utils.draw_landmarks(frame, hand, landmark_drawing_spec = landmark_drawing_spec)
            landmarks = hand.landmark

            index_tip = None
            thumb_tip = None
            middle_tip = None

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8:  # INDEX FINGER
                    cv2.circle(img=frame, center=(x, y), radius = 20, color=(0, 255, 0))
                    index_x_raw = screen_width / frame_width * x
                    index_y_raw = screen_height / frame_height * y

                    # SMOOTHING
                    index_x = int((1 - smoothing_factor) * index_x_raw + smoothing_factor * index_x_prev)
                    index_y = int((1 - smoothing_factor) * index_y_raw + smoothing_factor * index_y_prev)
                    pyautogui.moveTo(index_x, index_y)
                    index_x_prev, index_y_prev = index_x, index_y
                    index_tip = (index_x_raw, index_y_raw)

                if id == 4:  # THUMB
                    cv2.circle(img=frame, center=(x, y), radius = 30, color = (0, 255, 0))
                    thumb_x = screen_width / frame_width * x
                    thumb_y = screen_height / frame_height * y
                    thumb_tip = (thumb_x, thumb_y)

                if id == 12: # MIDDLE FINGER
                    cv2.circle(img=frame, center=(x, y), radius = 30, color = (0, 255, 0))
                    middle_x = screen_width / frame_width * x
                    middle_y = screen_height / frame_height * y
                    middle_tip = (middle_x, middle_y)

                if id == 20: #PINKY
                    cv2.circle(img=frame, center=(x, y), radius = 30, color = (0, 255, 0))
                    pinky_x = screen_width / frame_width * x
                    pinky_y = screen_height / frame_height * y
                    pinky_tip = (pinky_x, pinky_y)
                    

            # Left click if index and thumb pinch
            if index_tip and thumb_tip:
                distance = abs(index_tip[1] - thumb_tip[1])
                if distance < click_threshold:
                    print('Left click, :', distance)
                    pyautogui.click()
                    pyautogui.sleep(click_delay)


            # right click if middle and index touch
            if middle_tip and index_tip:
                distance = abs(middle_tip[1] - index_tip[1])
                if (distance < click_threshold):
                    print('Right click')
                    pyautogui.rightClick()
                    pyautogui.sleep(click_delay)

            #scroll down if pinky and thumb pinch
            if pinky_tip and thumb_tip:
                distance = abs(pinky_tip[1] - thumb_tip[1])
                if distance < click_threshold:
                    print("scrolling")
                    pyautogui.scroll(-1)
                    pyautogui.sleep(0.3)


    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()