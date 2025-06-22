import cv2
import mediapipe as mp
import pygame
import random
import math
import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gesture-Controlled Fruit Ninja")
clock = pygame.time.Clock()

# Load fruit image
fruit_img = pygame.image.load("fruit.png")
fruit_img = pygame.transform.scale(fruit_img, (60, 60))

# Font
font = pygame.font.SysFont(None, 40)

# Fruit class
class Fruit:
    def __init__(self):
        self.x = random.randint(100, WIDTH - 100)
        self.y = HEIGHT
        self.speed_y = random.randint(7, 12)
        self.sliced = False
        self.rect = fruit_img.get_rect(center=(self.x, self.y))

    def move(self):
        self.y -= self.speed_y
        self.rect.center = (self.x, self.y)

    def draw(self):
        if not self.sliced:
            screen.blit(fruit_img, self.rect)

# Webcam setup
cap = cv2.VideoCapture(0)

# Game state
fruits = []
spawn_timer = 0
score = 0
prev_points = []
max_points = 5  # store last 5 finger positions

running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Read webcam
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    current_point = None
    hand_open = False
    two_fingers = False
    palm_directions = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            thumb_tip = hand_landmarks.landmark[4]
            wrist = hand_landmarks.landmark[0]

            h, w, _ = frame.shape
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)
            current_point = (cx, cy)

            # Check for open hand gesture (all fingers extended)
            if (index_tip.y < hand_landmarks.landmark[6].y and
                middle_tip.y < hand_landmarks.landmark[10].y and
                ring_tip.y < hand_landmarks.landmark[14].y and
                pinky_tip.y < hand_landmarks.landmark[18].y and
                thumb_tip.x < hand_landmarks.landmark[3].x):
                hand_open = True

            # Check for peace sign gesture (only index and middle up)
            if (index_tip.y < hand_landmarks.landmark[6].y and
                middle_tip.y < hand_landmarks.landmark[10].y and
                ring_tip.y > hand_landmarks.landmark[14].y and
                pinky_tip.y > hand_landmarks.landmark[18].y):
                two_fingers = True

            # Add wrist movement to palm direction list
            palm_directions.append((int(wrist.x * w), int(wrist.y * h)))

    # Store current point in movement history
    if current_point:
        prev_points.append(current_point)
        if len(prev_points) > max_points:
            prev_points.pop(0)

    if len(palm_directions) > 5:
        palm_directions.pop(0)

    palm_swipe = False
    if len(palm_directions) >= 2:
        dx = palm_directions[-1][0] - palm_directions[0][0]
        dy = palm_directions[-1][1] - palm_directions[0][1]
        if abs(dx) > 50 or abs(dy) > 50:
            palm_swipe = True

    # Spawn fruit
    spawn_timer += 1
    if spawn_timer > 40:
        fruits.append(Fruit())
        spawn_timer = 0

    # Update fruit
    screen.fill((30, 30, 30))
    for fruit in fruits:
        fruit.move()
        fruit.draw()

        if hand_open and fruit.rect.collidepoint(current_point):
            fruit.sliced = True
            score += 1

        elif two_fingers and fruit.rect.collidepoint(current_point):
            fruit.sliced = True
            score += 1

        elif palm_swipe and fruit.rect.collidepoint(current_point):
            fruit.sliced = True
            score += 1

        elif current_point and len(prev_points) > 1:
            for i in range(len(prev_points) - 1):
                p1 = prev_points[i]
                p2 = prev_points[i + 1]
                if fruit.rect.clipline(p1, p2):
                    fruit.sliced = True
                    score += 1
                    break

    # Draw score
    score_text = font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(score_text, (10, 10))

    # Show game window
    pygame.display.flip()
    clock.tick(30)

    # Show webcam
    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.quit()
