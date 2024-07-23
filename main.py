from tkinter import Image
import cv2 
import mediapipe as mp
import numpy as np 

# inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5)

# inisialisasi MediaPipe Drawing 
mp_drawing = mp.solutions.drawing_utils

# inisialisasi kamera 
cap = cv2.VideoCapture(0)

def count_fingers(image, hand_landmarks, hand_connection):
    # Jari-jari dihitung berdasarkan posisi landmark
    thumb_tip = hand_landmarks.landmark[mp_hands.HandsLandmark.THUMB_TIP].y
    index_tip = hand_landmarks.landmark[mp_hands.HandsLandmark.INDEX_FINGER_TIP].y
    middle_tip = hand_landmarks.landmark[mp_hands.HandsLandmark.MIDDLE_FINGER_TIP].y
    ring_tip = hand_landmarks.landmark[mp_hands.HandsLandmark.RING_FINGER_TIP].y
    pinky_tip = hand_landmarks.landmark[mp_hands.HandsLandmark.PINKY_FINGER_TIP].y

    # Posisi MCP (Metacarpophalangeal) atau pangkal jari
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y

    # Menghitung jari yang terangkat 
    fingers = [
        thumb_tip > index_mcp,  
        index_tip > index_mcp,  
        middle_tip > middle_mcp,  
        ring_tip > ring_mcp,  
        pinky_tip > pinky_mcp 
    ]

    # Mengenali gestur
    if fingers == [True, False, False, False, False]:
        gesture = 'Peace'
    elif fingers == [True, False, False, False, False]:
        gesture = 'Thumbs Up'
    elif fingers == [False, True, True, False, False]:
        gesture = 'Victory'
    elif fingers == [True, False, False, False, True]:
        gesture = 'OK'
    elif fingers == [False, False, True, False, False]:
        gesture = 'Fck kata gw teh'
    else:
        gesture = 'Unknown'

    return gesture

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Mengabaikan frame kosong")
        continue

    # Buat background hitam 
    black_image = np.zeros_like(image)

    # Konversi gambar dari BGR ke RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Proses gambar dan deteksi tangan
    results = hands.process(image)

    # Tampilkan gesture yang dikenali
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                black_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = count_fingers(black_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(black_image, f'Gesture: {gesture}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('MediaPipe Hands', black_image)
            if cv2.waitKey(5) & 0xFF == ord('q'):  # Tekan q untuk keluar
                break

# Bebaskan sumber daya kamera dan tutup window
cap.release()
cv2.destroyAllWindows()



