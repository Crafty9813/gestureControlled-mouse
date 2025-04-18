import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
index_y_prev, index_x_prev = 0, 0
smoothing_factor = 0.7
click_threshold = 25
click_delay = 0.1 
movement_threshold = 50

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, ret = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks
    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark
            index_finger_tip = None
            thumb_tip = None

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8:  # index finger
                    cv2.circle(img=frame, center=(x, y), radius = 20, color=(0, 255, 0))
                    index_x_raw = screen_width / frame_width * x
                    index_y_raw = screen_height / frame_height * y

                    # smoothing
                    index_x = int((1 - smoothing_factor) * index_x_raw + smoothing_factor * index_x_prev)
                    index_y = int((1 - smoothing_factor) * index_y_raw + smoothing_factor * index_y_prev)
                    pyautogui.moveTo(index_x, index_y)
                    index_x_prev, index_y_prev = index_x, index_y
                    index_finger_tip = (index_x_raw, index_y_raw)

                if id == 4:  # thumb
                    cv2.circle(img=frame, center=(x, y), radius = 30, color = (0, 255, 255))
                    thumb_x = screen_width / frame_width * x
                    thumb_y = screen_height / frame_height * y
                    thumb_tip = (thumb_x, thumb_y)

                if id == 12: # middle finger
                    cv2.circle(img=frame, center=(x, y), radius = 30, color = (0, 255, 255))
                    middle_x = screen_width / frame_width * x
                    middle_y = screen_height / frame_height * y
                    middle_tip = (thumb_x, thumb_y)
                    

            # Left click if index and thumb pinch
            if index_finger_tip and thumb_tip:
                distance = abs(index_finger_tip[1] - thumb_tip[1])
                print('Distance btwn thumb & index finger:', distance)
                if distance < click_threshold:
                    pyautogui.click()
                    pyautogui.sleep(click_delay)


            # right click if middle and thumb pinch
            if middle_tip and thumb_tip:
                distance = abs(middle_tip[1] - thumb_tip[1])
                print('Distance btwn thumb & middle finger:', distance)
                if distance < click_threshold:
                    pyautogui.rightClick()
                    pyautogui.sleep(click_delay)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()