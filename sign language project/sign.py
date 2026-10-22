import cv2
import mediapipe as mp
import math

# =========================
# MediaPipe Setup
# =========================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# =========================
# Webcam
# =========================

cap = cv2.VideoCapture(0)

# =========================
# FULL SCREEN WINDOW
# =========================

cv2.namedWindow(
    "Hand Gesture Detection",
    cv2.WND_PROP_FULLSCREEN
)

cv2.setWindowProperty(
    "Hand Gesture Detection",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

# =========================
# Finger Detection Function
# =========================

def fingers_up(hand_landmarks):

    fingers = []

    # Thumb
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other Fingers
    tip_ids = [8, 12, 16, 20]

    for tip in tip_ids:

        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


# =========================
# Distance Function
# =========================

def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2


    )

# =========================
# Main Loop
# =========================

while True:

    success, frame = cap.read()

    if not success:
        break

    # Flip Frame
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process Hand
    result = hands.process(rgb)

    gesture = "Unknown"
    emoji = "❓"

    # =========================
    # Hand Detection
    # =========================

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            # Draw Landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            fingers = fingers_up(hand_landmarks)

            # =========================
            # ALPHABETS
            # =========================

            # A 🅰️
            if fingers == [0,0,0,1,1]:
                gesture = "A"
                emoji = "🅰️"

            # B 🅱️
            elif fingers == [0,1,1,1,1]:
                gesture = "B"
                emoji = "🅱️"

            # L 🫶
            elif fingers == [1,1,0,0,0]:
                gesture = "L"
                emoji = "🫶"

            # Y 💛
            elif fingers == [1,0,0,0,1]:
                gesture = "Y"
                emoji = "💛"

            # C ©️
            else:

                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]

                dist = distance(
                    thumb_tip,
                    index_tip
                )

                if 0.15 < dist < 0.35:
                    gesture = "C"
                    emoji = "©️"

            # =========================
            # GESTURES
            # =========================

            # STOP ✋
            if fingers == [1,1,1,1,1]:
                gesture = "STOP"
                emoji = "✋"

            # VICTORY ✌️
            elif fingers == [0,1,1,0,0]:
                gesture = "VICTORY"
                emoji = "✌️"

            # ROCK 🤘
            elif fingers == [0,1,0,0,1]:
                gesture = "ROCK"
                emoji = "🤘"

            # THUMBS UP 👍
            elif fingers == [1,0,0,0,0]:
                gesture = "THUMBS UP"
                emoji = "👍"

            # THUMBS DOWN 👎
            elif fingers == [0,0,0,0,1]:
                gesture = "THUMBS DOWN"
                emoji = "👎"

            # FIST ✊
            elif fingers == [0,0,0,0,0]:
                gesture = "FIST"
                emoji = "✊"

            # PEACE ☮️
            elif fingers == [0,1,1,1,0]:
                gesture = "PEACE"
                emoji = "☮️"

            # POINT UP ☝️
            elif fingers == [0,1,0,0,0]:
                gesture = "POINT UP"
                emoji = "☝️"

            # =========================
            # DISPLAY RESULT
            # =========================

            cv2.rectangle(
                frame,
                (10,10),
                (700,100),
                (0,0,0),
                -1
            )

            cv2.putText(
                frame,
                f"{gesture} {emoji}",
                (20,70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0,255,0),
                3
            )

    # =========================
    # SHOW OUTPUT
    # =========================

    cv2.imshow("Hand Gesture Detection", frame)

    # Press Q to Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# RELEASE
# =========================

cap.release()
cv2.destroyAllWindows()