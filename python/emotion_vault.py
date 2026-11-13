import cv2
import mediapipe as mp
import asyncio
import websockets
import json
import base64
import math
import torch
import torchvision.transforms as transforms
from PIL import Image

# 1. Initialize AI Modules
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

device = torch.device('cpu')
print(f"Loading TorchScript model mapped to: {device}")
emotion_model = torch.jit.load("emotion_model.pt", map_location=device)
emotion_model.eval()

# FIXED: Exact 5 classes from model_metadata.json
emotion_classes = ["Angry", "Happy", "Sad", "Surprise", "Neutral"]

# FIXED: 224x224 MobileNetV2 dimensions and ImageNet normalization
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

async def vision_server(websocket):
    print("Client connected to Phantom Vault.")
    
    cap = cv2.VideoCapture("/dev/video0")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    last_known_cursors = []
    last_known_emotion = "SCANNING..."
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            frame_count += 1
            
            # Process every 3rd frame to prevent CPU bottleneck
            if frame_count % 3 == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Hand Tracking
                hand_results = hands.process(rgb_frame)
                current_cursors = []
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        idx = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                        pinch_dist = math.hypot(idx.x - thumb.x, idx.y - thumb.y)
                        current_cursors.append({"x": idx.x, "y": idx.y, "is_grabbing": bool(pinch_dist < 0.05)})
                last_known_cursors = current_cursors

                # Emotion Detection
                current_emotion = "NO FACE"
                face_results = face_detection.process(rgb_frame)
                
                if face_results.detections:
                    detection = face_results.detections[0]
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    
                    raw_x = int(bboxC.xmin * w)
                    raw_y = int(bboxC.ymin * h)
                    raw_w = int(bboxC.width * w)
                    raw_h = int(bboxC.height * h)
                    
                    # 25% padding to capture eyebrows and chin
                    pad_w = int(raw_w * 0.25)
                    pad_h = int(raw_h * 0.25)
                    
                    x = max(0, raw_x - pad_w)
                    y = max(0, raw_y - pad_h)
                    x_end = min(w, raw_x + raw_w + pad_w)
                    y_end = min(h, raw_y + raw_h + pad_h)
                    
                    # Draw a targeting box on the UI stream so you can see what the AI sees
                    cv2.rectangle(frame, (x, y), (x_end, y_end), (0, 255, 255), 2)
                    
                    face_crop = frame[y:y_end, x:x_end]
                    
                    # Ensure crop is valid before passing to PyTorch
                    if face_crop.size != 0 and (x_end - x) > 30 and (y_end - y) > 30:
                        # FIXED: Convert directly to RGB (Model is MobileNetV2, not grayscale)
                        pil_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            output = emotion_model(input_tensor)
                            _, predicted = torch.max(output, 1)
                            current_emotion = emotion_classes[predicted.item()]
                
                last_known_emotion = current_emotion

            # Encode and Transmit
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            payload = {
                "cursors": last_known_cursors,
                "frame": frame_b64,
                "emotion": last_known_emotion
            }
            
            await websocket.send(json.dumps(payload))
            await asyncio.sleep(0.001)
            
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")
    finally:
        cap.release()

async def main():
    print("--> AI BACKEND running on ws://0.0.0.0:8765")
    start_server = websockets.serve(vision_server, "0.0.0.0", 8765)
    await start_server
    await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())