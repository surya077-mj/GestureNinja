import cv2
import mediapipe as mp
import pyautogui
import time
import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Webcam setup
cap = cv2.VideoCapture(0)

# Gesture tracking
swipe_history = []
gesture_cooldown = 0.4  # Faster detection
last_gesture_time = time.time()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    current_time = time.time()

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

        landmarks = handLms.landmark
        h, w, _ = img.shape
        wrist = landmarks[0]
        wrist_pos = (int(wrist.x * w), int(wrist.y * h))

        # Track swipe movement
        swipe_history.append(wrist_pos)
        if len(swipe_history) > 7:
            swipe_history.pop(0)

        # Detect gestures based on swipe
        if len(swipe_history) >= 2:
            dx = swipe_history[-1][0] - swipe_history[0][0]
            dy = swipe_history[-1][1] - swipe_history[0][1]

            if abs(dx) > 60 and current_time - last_gesture_time > gesture_cooldown:
                if dx > 0:
                    pyautogui.press("right")
                    print("👉 Swipe Right → Right Arrow")
                else:
                    pyautogui.press("left")
                    print("👈 Swipe Left → Left Arrow")
                last_gesture_time = current_time

            elif abs(dy) > 60 and current_time - last_gesture_time > gesture_cooldown:
                if dy > 0:
                    pyautogui.press("down")
                    print("👇 Swipe Down → Roll")
                else:
                    pyautogui.press("up")
                    print("☝️ Swipe Up → Jump")
                last_gesture_time = current_time

        # Index finger only → Move Left
        index_up = landmarks[8].y < landmarks[6].y
        middle_down = landmarks[12].y > landmarks[10].y
        ring_down = landmarks[16].y > landmarks[14].y
        pinky_down = landmarks[20].y > landmarks[18].y

        if index_up and middle_down and ring_down and pinky_down and current_time - last_gesture_time > gesture_cooldown:
            pyautogui.press("left")
            print("👆 Index Finger → Move Left")
            last_gesture_time = current_time

        # Peace sign → Move Right
        if (landmarks[8].y < landmarks[6].y and
            landmarks[12].y < landmarks[10].y and
            landmarks[16].y > landmarks[14].y and
            landmarks[20].y > landmarks[18].y and
            current_time - last_gesture_time > gesture_cooldown):
            pyautogui.press("right")
            print("✌️ Two Fingers → Move Right")
            last_gesture_time = current_time

        # Detect closed palm (jump)
        fingers_closed = []
        for tip_id in [8, 12, 16, 20]:
            tip = landmarks[tip_id]
            dip = landmarks[tip_id - 2]
            fingers_closed.append(tip.y > dip.y)

        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_closed = thumb_tip.x < thumb_ip.x if thumb_tip.x < 0.5 else thumb_tip.x > thumb_ip.x

        if all(fingers_closed) and not thumb_closed and current_time - last_gesture_time > gesture_cooldown:
            pyautogui.press("up")
            print("🤚 Palm Closed → Jump")
            last_gesture_time = current_time

        # Detect fist (roll)
        if all(fingers_closed) and thumb_closed and current_time - last_gesture_time > gesture_cooldown:
            pyautogui.press("down")
            print("✊ Fist → Roll")
            last_gesture_time = current_time

    cv2.imshow("Webcam Gesture Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
