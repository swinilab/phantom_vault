import cv2
import mediapipe as mp
import asyncio
import websockets
import json
import math
import torch
import torchvision.transforms as transforms
from PIL import Image

# Limit PyTorch thread usage to keep CPU free for game logic
torch.set_num_threads(2) 

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_face_detection = mp.solutions.face_detection
# Strict confidence threshold to eliminate background wall/chair false positives
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.82)

# Initialize PyTorch Emotion Model
device = torch.device('cpu')
print(f"Loading TorchScript model mapped to: {device}")
emotion_model = torch.jit.load("emotion_model.pt", map_location=device)
emotion_model.eval()

emotion_classes = ["Angry", "Happy", "Sad", "Surprise", "Neutral"]
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

async def vision_server(websocket):
    print("Client connected to Holographic Vault backend.")
    
    cap = cv2.VideoCapture("/dev/video0")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    last_known_cursors = []
    last_known_emotion = "NO FACE"
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            frame_count += 1
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # --- TICK: Hand Tracking (Every 2nd frame) ---
            if frame_count % 2 == 0:
                results = hands.process(rgb_frame)
                current_cursors = []
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        idx = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                        pinch_dist = math.hypot(idx.x - thumb.x, idx.y - thumb.y)
                        
                        current_cursors.append({
                            "x": idx.x,
                            "y": idx.y,
                            "is_grabbing": bool(pinch_dist < 0.05)
                        })
                last_known_cursors = current_cursors

            # --- TOCK: Emotion Detection (Every 7th frame) ---
            elif frame_count % 7 == 0:
                face_results = face_detection.process(rgb_frame)
                
                # Default strictly to NO FACE unless a high-confidence face box is verified
                detected_emotion = "NO FACE"
                
                if face_results.detections:
                    detection = face_results.detections[0]
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    
                    raw_x = int(bboxC.xmin * w)
                    raw_y = int(bboxC.ymin * h)
                    raw_w = int(bboxC.width * w)
                    raw_h = int(bboxC.height * h)
                    
                    pad_w = int(raw_w * 0.25)
                    pad_h = int(raw_h * 0.25)
                    
                    x = max(0, raw_x - pad_w)
                    y = max(0, raw_y - pad_h)
                    x_end = min(w, raw_x + raw_w + pad_w)
                    y_end = min(h, raw_y + raw_h + pad_h)
                    
                    face_crop = frame[y:y_end, x:x_end]
                    
                    if face_crop.size != 0:
                        if (x_end - x) < 80 or (y_end - y) < 80:
                            detected_emotion = "TOO FAR"
                        else:
                            cv2.rectangle(frame, (x, y), (x_end, y_end), (0, 255, 255), 1)
                            pil_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                            input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
                            
                            with torch.no_grad():
                                output = emotion_model(input_tensor)
                                _, predicted = torch.max(output, 1)
                                detected_emotion = emotion_classes[predicted.item()]
                
                last_known_emotion = detected_emotion

            # Encode frame to binary JPEG buffer
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            jpeg_bytes = buffer.tobytes()
            
            # Pack binary packet: [4 bytes header length] + [JSON Metadata] + [Raw JPEG Binary]
            metadata_json = json.dumps({
                "cursors": last_known_cursors,
                "emotion": last_known_emotion
            }).encode('utf-8')
            
            metadata_len = len(metadata_json)
            header = metadata_len.to_bytes(4, byteorder='big')
            
            await websocket.send(header + metadata_json + jpeg_bytes)
            await asyncio.sleep(0.01)
            
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")
    finally:
        cap.release()

async def main():
    print("Starting Ventuno Q High-Performance Vision Server on ws://localhost:8765...")
    start_server = websockets.serve(vision_server, "0.0.0.0", 8765)
    await start_server
    await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())