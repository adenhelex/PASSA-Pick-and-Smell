# PASSA-Pick-and-Smell

Local dashboard and controller for the PASSA pick-and-smell robot arm. The Mac runs the camera, YOLO-World detection, and the planner. The arm is an S2 RBAE02 Braccio driven through an Arduino Uno. Motion starts in dry-run. Arming is required before any UART command is sent.

## Run

Backend (needs camera permission, so do not sandbox it):

```bash
cd backend
.venv/bin/python -m app
```

Dashboard:

```bash
cd frontend
npm run dev
```

Dashboard: http://127.0.0.1:3000  
API: http://127.0.0.1:8000

After a backend restart, select the iPhone camera and the Ollama server again. The detector model is `backend/models/yolov8s-worldv2.onnx`. Rebuild it with `backend/scripts/export_yolo_world.py` after changing the class list in `backend/app/detect.py`.
