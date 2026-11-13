import cv2
import mediapipe as mp
import asyncio
import websockets
import json
import base64
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1)

cap = cv2.VideoCapture(0)

async def process_camera(websocket):
    print("UI Client connected! Streaming face, hands, and video...")
    last_gesture_time = time.time()
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            await asyncio.sleep(0.01)
            continue
            
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 1. Process Face
        face_results = face_mesh.process(rgb_image)
        state = "neutral"
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                top_lip = face_landmarks.landmark[13].y
                bottom_lip = face_landmarks.landmark[14].y
                if (bottom_lip - top_lip) > 0.06:
                    state = "surprise"

        # 2. Process Hands (Finger Counting)
        hand_results = hands.process(rgb_image)
        detected_gesture = None
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Compare fingertips (8,12,16,20) to their lower joints (6,10,14,18)
                tips = [8, 12, 16, 20]
                joints = [6, 10, 14, 18]
                fingers_up = 0
                
                for tip, joint in zip(tips, joints):
                    if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[joint].y:
                        fingers_up += 1
                        
                # 1.5-second cooldown between inputs
                if time.time() - last_gesture_time > 1.5: 
                    if fingers_up == 1:
                        detected_gesture = "1"
                    elif fingers_up == 2:
                        detected_gesture = "2"
                    elif fingers_up == 3:
                        detected_gesture = "3"
                        
                    if detected_gesture:
                        last_gesture_time = time.time()
                        break

        # 3. Compress and Transmit
        small_frame = cv2.resize(image, (320, 240))
        _, buffer = cv2.imencode('.jpg', small_frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        payload = {"affect": state, "frame": frame_b64}
        if detected_gesture:
            payload["gesture"] = detected_gesture
            
        try:
            await websocket.send(json.dumps(payload))
        except websockets.exceptions.ConnectionClosed:
            print("UI Client disconnected.")
            break
            
        await asyncio.sleep(0.05) 

async def main():
    async with websockets.serve(process_camera, "localhost", 8765):
        print("WebSocket bridge active on ws://localhost:8765")
        await asyncio.Future() 

if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        cap.release()