# 😀 PhantomVault by Thant Zin Moe
🪞 Phantom Vault: Smart Mirror & Emotion AI

Phantom Vault is an interactive, gamified smart mirror application designed for the Ventuno Q (ARM-based edge device). It merges dual-hand gesture tracking with real-time facial emotion detection, originally adapted from the RosMaster X3 robotics platform.

The system requires users to physically interact with holographic UI elements through a 3-stage "Firewall Breach" while continuously monitoring their biometric emotional state.

✨ Features

Real-Time Emotion Telemetry: Detects 5 emotional states (Happy, Sad, Angry, Surprise, Neutral) using a PyTorch MobileNetV2 classifier.

Dual-Hand Spatial Tracking: Powered by MediaPipe, allowing users to "pinch and grab" virtual elements on the mirror.

High-Performance Edge AI:

Tick-Tock Async Scheduling: Hand tracking and face detection are decoupled across alternate frames to prevent ARM CPU throttling.

Dual-Resolution Decoupling: AI crunches on a lightweight 640x480 matrix while the frontend renders a GPU-accelerated fullscreen UI.

Binary WebSocket Streaming: Zero-lag transmission using raw ArrayBuffer bytes rather than base64 strings.

Seamless Soft-Reset: Auto-restarts the gamification loop without dropping the Linux Video4Linux (V4L2) camera buffer.

🎮 The 3-Stage Gamification Loop

Stage 1 (Pattern Lock): One-handed pinch-to-draw gesture to connect a 3x3 node grid matching a secret security code.

Stage 2 (Key Override): Dual-hand tracking to simultaneously grab and separate two override keys into designated drop zones.

Stage 3 (Core Compression): Two-handed pinch-and-compress gesture to stabilize an oscillating "unstable core" within a strict pixel tolerance.

🛠️ Architecture & Tech Stack

Hardware: Ventuno Q (ARM architecture), USB Web Camera

Backend: Python 3.12

OpenCV (Camera capture and frame processing)

MediaPipe (Facial bounding boxes and hand landmark detection)

PyTorch / TorchScript (Emotion classification inference)

websockets (Async binary data transmission)

Frontend: HTML5, CSS3, Vanilla JavaScript (Browser-based UI, Canvas drawing, ArrayBuffer rendering)

🚀 Installation & Setup

Prerequisites

A Linux/ARM environment (e.g., Ventuno Q) with a connected camera (/dev/video0).

Python 3.10+ installed.

1. Clone the Repository

git clone https://github.com/swinilab/phantom-vault.git
cd phantom-vault


2. Set Up the Virtual Environment

It is highly recommended to run the AI models inside an isolated virtual environment.

python3 -m venv vault_env
source vault_env/bin/activate


3. Install Dependencies

pip install opencv-python mediapipe websockets torch torchvision pillow


(Note: Ensure you have the emotion_model.pt TorchScript model file in the root directory).

4. Launch the Vision Server

Start the Python backend to boot the camera and AI models.

python3 game_emotion_vault.py


5. Open the Interface

Open game_emotion_index.html in any modern web browser (e.g., Chromium) in full-screen mode to begin the interactive experience.

🧠 Smart Fallbacks & Error Handling

NO FACE: If the user steps out of frame, the UI cleanly defaults to a deactivated state, preventing "ghost" background detections.

TOO FAR: Distance gating ensures the PyTorch model only classifies faces when they meet a 80x80 pixel minimum bounding box size to maintain accuracy.

👨‍💻 Credits

Developed and optimized by Thant Zin Moe (IT Lecturer & Researcher, Swinburne Vietnam) for the Swinburne Innovation Lab.



