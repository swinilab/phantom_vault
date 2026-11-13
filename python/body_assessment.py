import cv2
import mediapipe as mp
import asyncio
import websockets
import json
import base64
import math
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

async def process_camera(websocket):
    print("Smart Mirror connected! High-Accuracy Body Scanner active...")
    state = "ALIGN"
    compliment = ""
    lock_time = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            await asyncio.sleep(0.01)
            continue
            
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_image)
        
        # Get actual pixel dimensions of the camera frame
        ih, iw, _ = image.shape
        
        ui_message = "STEP INTO THE FRAME"
        is_aligned = False
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            nose = landmarks[0]
            l_sh, r_sh = landmarks[11], landmarks[12]
            l_hip, r_hip = landmarks[23], landmarks[24]
            l_ank, r_ank = landmarks[27], landmarks[28]
            
            # Check visibility
            if (nose.visibility > 0.6 and l_ank.visibility > 0.6 and r_ank.visibility > 0.6):
                if 0.35 < nose.x < 0.65:
                    is_aligned = True
                    
            if is_aligned:
                if state == "ALIGN":
                    state = "MEASURING"
                    lock_time = time.time()
                    ui_message = "HOLD STILL... SCANNING"
                elif state == "MEASURING":
                    if time.time() - lock_time > 2.0:
                        # 1. Convert to true pixel coordinates to fix aspect ratio distortion
                        sh_width_px = math.dist([l_sh.x * iw, l_sh.y * ih], [r_sh.x * iw, r_sh.y * ih])
                        hip_width_px = math.dist([l_hip.x * iw, l_hip.y * ih], [r_hip.x * iw, r_hip.y * ih])
                        
                        # 2. Calculate true height in pixels
                        head_y_px = nose.y * ih
                        ankle_y_px = ((l_ank.y + r_ank.y) / 2) * ih
                        total_height_px = abs(ankle_y_px - head_y_px)
                        
                        # 3. New Advanced Formula: (Average of Shoulders & Hips) / Total Height
                        avg_body_width = (sh_width_px + hip_width_px) / 2
                        ratio = avg_body_width / total_height_px
                        
                        print(f"--- SCAN COMPLETE ---")
                        print(f"True Body Ratio: {ratio:.3f}")
                        
                        # Calibrated Thresholds
                        if ratio > 0.265:
                            compliment = "A little fat, but very beautiful! 💖"
                        elif ratio < 0.220:
                            compliment = "Very thin, need to eat! But very pretty! 🍔✨"
                        else:
                            compliment = "Very beautiful, perfect proportions! 🌟"
                        
                        state = "RESULT"
                        ui_message = compliment
                    else:
                        ui_message = "HOLD STILL... SCANNING"
                elif state == "RESULT":
                    ui_message = compliment
                    if time.time() - lock_time > 7.0: 
                        state = "ALIGN"
            else:
                state = "ALIGN"
                ui_message = "FIT FULL BODY INSIDE THE OUTLINE"
        else:
            state = "ALIGN"
            ui_message = "STEP INTO THE FRAME"

        small_frame = cv2.resize(image, (640, 480))
        _, buffer = cv2.imencode('.jpg', small_frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        try:
            await websocket.send(json.dumps({
                "frame": frame_b64,
                "message": ui_message,
                "state": state
            }))
        except websockets.exceptions.ConnectionClosed:
            break
            
        await asyncio.sleep(0.05) 

async def main():
    async with websockets.serve(process_camera, "localhost", 8765):
        await asyncio.Future() 

if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        cap.release()