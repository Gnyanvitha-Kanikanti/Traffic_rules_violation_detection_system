Traffic Rules Violation Detection System
This project is an AI-powered real-time detection system for identifying common traffic rule violations such as red light jumping, helmetless riding, and license plate recognition using computer vision and deep learning models like YOLOv8 and Roboflow Inference API.

Key Features
Red Light Violation Detection
Detects vehicles crossing a red light using dynamic ROI selection and traffic light state recognition.
Helmet Detection
Uses the Roboflow API to detect whether motorbike riders are wearing helmets or not.
License Plate Detection and OCR
Detects vehicle number plates and extracts text using YOLO and Tesseract OCR.
Real-Time Video Processing
Works with video input (.mp4) to process and display live results with bounding boxes and violation messages.
Dynamic ROI Selection
Allows users to manually select the area of interest in a frame to monitor for violations.

Tech Stack
Language: Python
Libraries: OpenCV, NumPy, pytesseract, re, uuid
Models:YOLOv8 for vehicle, traffic light, and plate detection
Roboflow Hosted Model for helmet detection
OCR: Tesseract OCR

project/
│
├── main.py                     
├── yolov8m.pt     # https://github.com/ultralytics/assets/releases/tag/v8.3.0            
├── license_plate_detector.pt                     
└── README.md  

Output
The system will display:
Bounding boxes with labels and confidence
Warnings if a vehicle violates red light
Helmet violations in a different color
Number plate text in a separate window
