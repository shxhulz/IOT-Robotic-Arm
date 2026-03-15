# IoT Robotic Arm v2: Smart Waste Sorting with Real-Time Analytics

An intelligent robotic arm system that uses computer vision and machine learning to automatically detect, pick up, and sort waste materials. **Version 2** introduces a scalable event-driven architecture with **Kafka**, a **FastAPI** backend, and a modern **React** dashboard for real-time monitoring and analytics.

## 🌟 New in v2

-   **Event-Driven Architecture**: Decoupled communication using Apache Kafka.
-   **Real-Time Dashboard**: React + TypeScript + Shadcn UI dashboard for live monitoring.
-   **Analytics**: Historical data tracking (SQLite) with interactive charts.
-   **Resilience**: "Safe Producer" design ensures the robot operates even if the network fails.
-   **Scalability**: Ready for multi-robot management.

## 🏗️ System Architecture v2

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Robot Arm   │───▶│    Kafka     │───▶│   Backend    │
│ (Controller) │    │   (Broker)   │    │  (FastAPI)   │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Hardware   │    │  Zookeeper   │    │   Frontend   │
│ (Arduino/Cam)│    │              │    │   (React)    │
└──────────────┘    └──────────────┘    └──────────────┘
```

## 📚 Technical Documentation & Code Explanation

### 1. Robotic Arm Controller (`src/controller/`)
The core logic that drives the physical hardware.
-   **`robotic_arm_controller.py`**: The central orchestrator.
    -   **`RoboticArmController` Class**: Manages the state machine (Scanning -> Detecting -> Centering -> Picking -> Sorting).
    -   **`SafeProducer` Class**: A custom wrapper around `KafkaProducer`. It wraps `producer.send()` in a `try-except` block. This is critical for resilience: if the network fails or Kafka is down, the robot logs the error but **continues to operate physically**.
    -   **Event Triggers**:
        -   `detection`: Fired when YOLO confidence exceeds `CONFIDENCE_THRESHOLD`.
        -   `pickup_success`: Fired after the arm successfully releases the object in the bin.
        -   `pickup_fail`: Fired if the object is lost during movement.

### 2. Dashboard Backend (`dashboard_backend/`)
The bridge between the hardware and the UI, built with **FastAPI**.
-   **`main.py`**:
    -   **Lifespan Events**: Uses `asynccontextmanager` to initialize the SQLite database and start the Kafka consumer background task when the server boots.
    -   **`/ws` Endpoint**: Handles WebSocket connections. It uses a `ConnectionManager` to broadcast Kafka messages to all connected frontend clients instantly.
    -   **`/stats/history` Endpoint**: Serves hourly trends. It uses `datetime.now(timezone.utc)` to ensure accurate time bucketing, preventing timezone mismatches (e.g., showing 10 AM instead of 4 PM).
-   **`consumer.py`**:
    -   **Async Consumption**: Uses `aiokafka` to consume messages from `robot_events` and `robot_telemetry` topics without blocking the main thread.
    -   **Data Persistence**: Critical events (pickups, detections) are saved to `robot_analytics.db` via SQLAlchemy.
    -   **Real-time Broadcast**: Immediately pushes incoming data to the WebSocket manager.

### 3. Dashboard Frontend (`dashboard_frontend/`)
A modern, responsive UI built with **React**, **TypeScript**, and **Shadcn UI**.
-   **`App.tsx`**:
    -   **WebSocket Integration**: Opens a persistent connection to the backend. Updates the `stats` (counters) and `logs` (activity feed) state variables in real-time upon receiving messages.
    -   **System Status**: Automatically detects connection loss and updates the "System Online/Offline" indicator.
-   **`components/LineChartInteractive.tsx`**:
    -   **Dynamic Visualization**: Uses `recharts` to render a 3-line chart (Plastic, Metal, Paper).
    -   **Smart Formatting**: Receives UTC timestamps from the backend but formats them to the user's **Local Time** on the X-axis using `toLocaleTimeString()`.
-   **Styling (`index.css`)**:
    -   **Dark Glassmorphism**: Implements a "frosted glass" aesthetic using `backdrop-filter: blur()` and semi-transparent RGBA backgrounds, creating a premium, futuristic look.

## 🔧 Hardware Requirements

### Essential Components

-   **Arduino UNO** - Main microcontroller
-   **Adafruit PWM Servo Driver (PCA9685)** - Controls up to 16 servos
-   **6x Servo Motors** - For robotic arm joints and gripper
-   **HC-SR04 Ultrasonic Sensor** - Distance measurement
-   **ESP 32 Camera** - Object detection
-   **Power Supply (5V 10A)** - External power for servos

## 💻 Software Requirements

### Core
-   **Python 3.10+**
-   **Docker & Docker Compose** (for Kafka)
-   **Node.js 18+** (for Dashboard)

### Python Dependencies
-   `opencv-python`, `ultralytics`, `pyserial` (Controller)
-   `fastapi`, `uvicorn`, `sqlalchemy`, `aiokafka` (Backend)
-   `kafka-python` (Shared)

## 🚀 Installation & Setup

### 1. Infrastructure (Kafka)
Start the message broker:
```bash
cd d:\Robotic_Arm_v2
docker-compose up -d
```

### 2. Backend (API & Consumer)
```bash
cd d:\Robotic_Arm_v2\dashboard_backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
*Runs on http://localhost:8000*

### 3. Frontend (Dashboard)
```bash
cd d:\Robotic_Arm_v2\dashboard_frontend
npm install
npm run dev
```
*Accessible at http://localhost:5173*

### 4. Robotic Arm Controller
```bash
cd d:\Robotic_Arm_v2\IoT-Robotic-Arm
pip install -r requirements.txt
python src/main.py
```

## 📊 Dashboard Features

1.  **Live Status**: Monitor Camera and Controller connectivity.
2.  **Real-Time Counters**: Track sorted items (Paper, Metal, Plastic) instantly.
3.  **Interactive Charts**: View hourly sorting trends over the last 24 hours.
4.  **Telemetry**: Live distance sensor readings from the robot.
5.  **Activity Log**: A scrolling feed of every action taken by the robot.

## ⚙️ Configuration

### Controller Config (`src/config/config.py`)
-   **CAMERA_STREAM_URL**: URL for the IP camera stream.
-   **SERVO_PORT**: COM port for Arduino (e.g., `COM3`).
-   **CONFIDENCE_THRESHOLD**: YOLO detection sensitivity.
-   **CENTER_THRESHOLD**: Pixel tolerance used to treat a target as centered.
-   **CENTERING_OFFSET_X**: Pixel offset that shifts the desired visual centering point left/right to match gripper calibration.
-   **CENTERING_STEP_DEGREES**: Base-servo correction size used during centering.
-   **CENTERING_SETTLE_SECONDS**: Delay before the post-centering camera recheck.
-   **CENTERING_MAX_ADJUSTMENTS**: Safety limit for centering retries per pickup task.

### Dashboard Config
-   **Backend**: `dashboard_backend/main.py` (Port 8000)
-   **Frontend**: `dashboard_frontend/vite.config.ts` (Port 5173)

## 🤝 Contributing

1.  Fork the repository
2.  Create a feature branch
3.  Commit your changes
4.  Push to the branch
5.  Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

---
**Built with ❤️ by Nevin-A-S**
