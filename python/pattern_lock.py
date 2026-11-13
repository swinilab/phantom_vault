import cv2
import mediapipe as mp
import asyncio
import websockets
import json
import base64
import math

# Initialize MediaPipe Hands (Lowered confidence for Stage 3 occlusion stability)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

async def vision_server(websocket):
    print("Client connected to Holographic Vault backend.")
    
    # 1. FORCE 1080p WEBCAM RESOLUTION (The Sweet Spot)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    frame_count = 0
    last_known_cursors = []
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # FIX: Flip the frame horizontally to fix the mirroring issue
            frame = cv2.flip(frame, 1)
                
            frame_count += 1
            
            # OPTIMIZATION: Run heavy AI inference only on even frames
            if frame_count % 2 == 0:
                # OPTIMIZATION: Downscale the frame JUST for MediaPipe AI (640x480)
                small_frame = cv2.resize(frame, (640, 480))
                rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                results = hands.process(rgb_frame)
                current_cursors = []
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Index finger tip (Landmark 8) and Thumb tip (Landmark 4)
                        idx = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                        
                        # Calculate pinch distance for the "is_grabbing" trigger
                        pinch_dist = math.hypot(idx.x - thumb.x, idx.y - thumb.y)
                        is_grabbing = bool(pinch_dist < 0.05)
                        
                        current_cursors.append({
                            "x": idx.x,
                            "y": idx.y,
                            "is_grabbing": is_grabbing
                        })
                
                # Update the cache with fresh AI data
                last_known_cursors = current_cursors

            # 2. ENCODE FULL HD VIDEO FEED: Send original frame with 85% JPEG compression
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # 3. BROADCAST: Send payload every single frame
            payload = {
                "cursors": last_known_cursors,
                "frame": frame_b64
            }
            
            await websocket.send(json.dumps(payload))
            
            # Yield control briefly to the async event loop to maintain WebSocket stability
            await asyncio.sleep(0.001)
            
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")
    finally:
        cap.release()

async def main():
    print("Starting Ventuno Q Vision Server on ws://localhost:8765...")
    start_server = websockets.serve(vision_server, "0.0.0.0", 8765)
    await start_server
    await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())