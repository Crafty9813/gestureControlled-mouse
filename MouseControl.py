import cv2
import mediapipe as mp
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

screen_width, screen_height = pyautogui.size()

mouse_sensitivity = 2

scroll_sensitivity = 5

finger_tip_threshold = 0.03

def get_finger_state(hand_landmarks):
    if hand_landmarks:
        landmarks = hand_landmarks.landmark
        thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

        thumb_base = landmarks[mp_hands.HandLandmark.THUMB_CMC]
        index_base = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_base = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_base = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_base = landmarks[mp_hands.HandLandmark.PINKY_MCP]

        fingers = {
            'thumb': thumb_tip.y < thumb_base.y - finger_tip_threshold,
            'index': index_tip.y < index_base.y - finger_tip_threshold,
            'middle': middle_tip.y < middle_base.y - finger_tip_threshold,
            'ring': ring_tip.y < ring_base.y - finger_tip_threshold,
            'pinky': pinky_tip.y < pinky_base.y - finger_tip_threshold,
        }
        return fingers
    return None

def get_hand_center(hand_landmarks, image_width, image_height):
    #Calculates center of detected hand
    if hand_landmarks:
        x_sum = 0
        y_sum = 0
        for landmark in hand_landmarks.landmark:
            x_sum += landmark.x
            y_sum += landmark.y
        return int(x_sum / len(hand_landmarks.landmark) * image_width), \
               int(y_sum / len(hand_landmarks.landmark) * image_height)
    return None, None

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:
    previous_hand_center = None
    scrolling = False
    scroll_start_y = None

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("No camera frame!")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = image.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                finger_state = get_finger_state(hand_landmarks)
                hand_center_x, hand_center_y = get_hand_center(hand_landmarks, image_width, image_height)

                if finger_state:
                    # Mouse (index finger)
                    if finger_state['index'] and not finger_state['middle'] and not finger_state['ring'] and not finger_state['pinky'] and finger_state['thumb']:
                        if previous_hand_center and hand_center_x is not None and hand_center_y is not None:
                            delta_x = (hand_center_x - previous_hand_center[0]) * mouse_sensitivity
                            delta_y = (hand_center_y - previous_hand_center[1]) * mouse_sensitivity
                            pyautogui.move(delta_x, delta_y)
                        previous_hand_center = (hand_center_x, hand_center_y)
                        scrolling = False

                    # Left click (fist)
                    elif not finger_state['index'] and not finger_state['middle'] and not finger_state['ring'] and not finger_state['pinky'] and finger_state['thumb']:
                        pyautogui.click()
                        print("Left Click")

                    # Right click (ring finger)
                    elif not finger_state['index'] and not finger_state['middle'] and finger_state['ring'] and not finger_state['pinky'] and finger_state['thumb']:
                        pyautogui.rightClick()
                        print("Right Click")

                    # Scroll (index and middle fingers up)
                    elif finger_state['index'] and finger_state['middle'] and not finger_state['ring'] and not finger_state['pinky'] and finger_state['thumb']:
                        if hand_center_y is not None:
                            if not scrolling:
                                scrolling = True
                                scroll_start_y = hand_center_y
                            else:
                                if scroll_start_y is not None:
                                    delta_y = (hand_center_y - scroll_start_y)
                                    if abs(delta_y) > 20:
                                        scroll_amount = int(delta_y / scroll_sensitivity)
                                        pyautogui.scroll(-scroll_amount)
                                        scroll_start_y = hand_center_y
                                        print(f"Scroll: {scroll_amount}")
                        previous_hand_center = None # Reset mouse

                    else:
                        previous_hand_center = None
                        scrolling = False
                        scroll_start_y = None

        else:
            previous_hand_center = None
            scrolling = False
            scroll_start_y = None

        cv2.imshow('frame', cv2.flip(image, 1))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()