# Working Activity Monitoring System

## Overview
This project implements a real-time idle person detection system using YOLO (You Only Look Once) object detection framework. The system tracks people within a defined Region of Interest (ROI) and detects whether they are idle (not moving) for a configurable period of time.

## Features
- Person detection and tracking using YOLO
- Region of Interest (ROI) support for targeted detection
- Idle person detection with configurable thresholds
- Support for multiple video sources (webcam, video files, RTSP streams)
- Optimized threaded RTSP stream handling for improved performance
- Real-time visualization of detection results
- Comprehensive logging system
- Configurable parameters through a config file
- Windows startup script for automatic execution

## Requirements
- torch==2.0.1+cu118
- torchvision==0.15.2+cu118
- numpy==1.26.4
- ultralytics==8.3.111
- onnx==1.17.0
- onnxruntime==1.19.2
- onnxruntime-gpu==1.18.1


## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/idle-detection-system.git
cd idle-detection-system

# Install dependencies
pip install -r requirements.txt
```

## Configuration
The system is configured using a config file. An example configuration:

```
"project_name" : "Vision Assignment",
"author": "Prakash pacharne",
"mode" : "video",
"image_size" : 640,
"classes" : 0,
"persist" : "True",
"conf_threshold" : 0.2,
"resize_width" : 640,
"resize_height": 640,
"normalize_frames" : "True",
"log_file_path" : "C:\\Users\\ASUS\\OneDrive\\Desktop\\worker\\assets\\idle_detection.log",
"pretrained_model_pt_path" : "C:\\Users\\ASUS\\OneDrive\\Desktop\\worker\\assets\\yolov9t.pt",
"pretrained_model_onnx_path" : "C:\\Users\\ASUS\\OneDrive\\Desktop\\worker\\assets\\yolov9t.onnx",
"rtsp_link" : "rtsp://admin:admin123@192.168.1.111:554//Streaming//Channel//101",
"inference_video" : "C:\\Users\\ASUS\\OneDrive\\Desktop\\worker\\assets\\sleeping.webm",
"roi_points" : "C:\\Users\\ASUS\\OneDrive\\Desktop\\worker\\assets\\roi_points.txt",
"verbose_history" : 0,
"iou_threshold": 0.3,
"dis_lines" : "y",
"idle_alert_threshold" : 10,
"movement_threshold" : 2.5
```

## ROI Configuration
The Region of Interest (ROI) can be defined in a text file with four corner points. Each line should contain x,y coordinates:

```
100,100
540,100
540,380
100,380
```

## Usage
```bash
python main.py
```

### Windows Startup Script
The repository includes a batch file (`Elansol Vision Application.bat`) that can be used to automatically start the application in the background:

```batch
@echo off
start /b "Task Scheduler by Prakash Pacharne" /min "C:\Users\ASUS\AppData\Local\Programs\Python\Python39\python.exe" "C:\Users\ASUS\OneDrive\Desktop\worker\main.pyw"
```

To use this script:
1. Modify the paths to match your Python installation and project location
2. Open the shell:startup using windows + R.
3. Place the batch file in this Windows startup folder.

### Controls
- Press 'q' to quit the application

## How It Works
1. The system loads the configuration and initializes the YOLO model
2. Video frames are captured from the specified source:
   - For local videos and webcams: Standard OpenCV VideoCapture
   - For RTSP streams: Custom threaded VideoCapture implementation
3. Each frame is processed:
   - Resized and normalized (if enabled)
   - ROI mask is applied
   - YOLO detection is performed
   - People are tracked across frames
   - Movement analysis determines if a person is idle
   - Visual indicators show idle status
4. Results are displayed in real-time

## RTSP Implementation
The updated code includes a specialized threaded VideoCapture class for RTSP streams that:
- Runs frame capture in a separate thread to prevent blocking
- Automatically reconnects if the connection is lost
- Uses a queue system to manage frames efficiently
- Prevents frame lag by discarding outdated frames
- Improves overall stability and performance for network camera streams

## Detection Logic
- A person is considered idle if they don't move more than the `movement_threshold` for longer than the `idle_alert_threshold` time
- Only people within the defined ROI are tracked and analyzed
- Each tracked person has an individual idle timer

## Visualization
- Green bounding boxes: Moving people
- Red bounding boxes: Idle people
- The display shows:
  - FPS (Frames Per Second)
  - Number of people in ROI
  - Number of idle people
  - Frame size and normalization status

## Error Handling
The system includes comprehensive error handling and logging to ensure reliable operation even in edge cases. The RTSP implementation has additional error recovery mechanisms for network disruptions.

## License
[Developed & Managed By: Prakash Pacharne]
