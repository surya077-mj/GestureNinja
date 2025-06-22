import cv2
import mediapipe as mp
import pyautogui
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
swipe_history = []
gesture_cooldown = 0.6  # Slightly longer cooldown
gesture_sensitivity = 80
last_gesture_time = time.time()
executed_gesture = None
frame_count = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    display_img = img.copy()

    process_img = cv2.resize(img, (320, 240))
    img_rgb = cv2.cvtColor(process_img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    current_time = time.time()

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(display_img, handLms, mp_hands.HAND_CONNECTIONS)

        landmarks = handLms.landmark
        h, w, _ = process_img.shape
        wrist = landmarks[0]
        wrist_pos = (int(wrist.x * display_img.shape[1]), int(wrist.y * display_img.shape[0]))
        swipe_history.append(wrist_pos)
        if len(swipe_history) > 5:
            swipe_history.pop(0)

        dx = swipe_history[-1][0] - swipe_history[0][0]
        dy = swipe_history[-1][1] - swipe_history[0][1]

        if abs(dx) > gesture_sensitivity and current_time - last_gesture_time > gesture_cooldown:
            direction = "right" if dx > 0 else "left"
            if executed_gesture != direction:
                pyautogui.press(direction)
                print("👉 Swipe Right" if dx > 0 else "👈 Swipe Left")
                last_gesture_time = current_time
                executed_gesture = direction

        elif abs(dy) > gesture_sensitivity and current_time - last_gesture_time > gesture_cooldown:
            direction = "down" if dy > 0 else "up"
            if executed_gesture != direction:
                pyautogui.press(direction)
                print("👇 Swipe Down" if dy > 0 else "☝️ Swipe Up")
                last_gesture_time = current_time
                executed_gesture = direction

        def is_finger_up(tip, pip):
            return landmarks[tip].y < landmarks[pip].y - 0.02

        index_up = is_finger_up(8, 6)
        middle_up = is_finger_up(12, 10)
        ring_down = not is_finger_up(16, 14)
        pinky_down = not is_finger_up(20, 18)

        if index_up and not middle_up and ring_down and pinky_down and current_time - last_gesture_time > gesture_cooldown:
            if executed_gesture != "index_left":
                pyautogui.press("left")
                print("👆 Index → Left")
                last_gesture_time = current_time
                executed_gesture = "index_left"

        if index_up and middle_up and ring_down and pinky_down and current_time - last_gesture_time > gesture_cooldown:
            if executed_gesture != "peace_right":
                pyautogui.press("right")
                print("✌️ Two Fingers → Right")
                last_gesture_time = current_time
                executed_gesture = "peace_right"

        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_extended = thumb_tip.x - thumb_ip.x > 0.08
        fingers_folded = all(landmarks[tip].y > landmarks[tip - 2].y + 0.02 for tip in [8, 12, 16, 20])

        if thumb_extended and fingers_folded and current_time - last_gesture_time > gesture_cooldown:
            if executed_gesture != "thumb_right":
                pyautogui.press("right")
                print("👍 Thumb → Right")
                last_gesture_time = current_time
                executed_gesture = "thumb_right"

        fingers_closed = [landmarks[tip].y > landmarks[tip - 2].y + 0.02 for tip in [8, 12, 16, 20]]
        thumb_closed = thumb_tip.x < thumb_ip.x if thumb_tip.x < 0.5 else thumb_tip.x > thumb_ip.x

        if all(fingers_closed) and not thumb_closed and current_time - last_gesture_time > gesture_cooldown:
            if executed_gesture != "palm_jump":
                pyautogui.press("up")
                print("🖐️ Palm Closed → Jump")
                last_gesture_time = current_time
                executed_gesture = "palm_jump"

        if all(fingers_closed) and thumb_closed and current_time - last_gesture_time > gesture_cooldown:
            if executed_gesture != "fist_roll":
                pyautogui.press("down")
                print("✊ Fist → Roll")
                last_gesture_time = current_time
                executed_gesture = "fist_roll"

    else:
        executed_gesture = None  # Reset if no hand detected

    frame_count += 1
    cv2.imshow("Webcam Gesture Control", display_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
