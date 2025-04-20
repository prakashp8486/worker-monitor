import sys
sys.dont_write_bytecode = True

import cv2
import time
import numpy as np
import traceback
import logging
from assets.config_reader import config_reader
from assets.videocaptureclass import VideoCapture

try:
    data = config_reader()

    project_name = data["project_name"]
    author_name = data["author"]
    mode = data['mode']
    image_size = int(data['image_size'])
    classes = int(data['classes'])
    persist = data['persist'] == "True"
    conf_threshold = float(data['conf_threshold'])
    log_file_path = data['log_file_path']
    model_path_pt = data['pretrained_model_pt_path']
    model_path_onnx = data['pretrained_model_onnx_path']
    rtsp_link = data['rtsp_link']
    video_path = data['inference_video']
    roi_file_path = data['roi_points']
    verbose_history = int(data['verbose_history'])
    iou_threshold = float(data['iou_threshold'])
    dis_lines = data['dis_lines']
    idle_alert_threshold = float(data['idle_alert_threshold'])
    movement_threshold = float(data['movement_threshold'])
    resize_width =  int(data['resize_width'])
    resize_height = int(data['resize_height'])
    normalize_frames = data['normalize_frames'] == 'True'

    default_roi_points = [(100, 100), (540, 100), (540, 380), (100, 380)]
    
except Exception as e:
    print(f"Error reading config: {e}")
    sys.exit(1)

try:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
except Exception as e:
    print(f"Warning: Could not configure logging: {e}")

try:
    from ultralytics import YOLO
except ImportError as e:
    logging.error(f"Failed to import YOLO: {e}")
    print("Please install ultralytics: pip install ultralytics")
    sys.exit(1)

roi_points = []
roi_selected = False

def point_in_polygon(point, polygon):
    try:
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            x_intersect = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= x_intersect:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    except Exception as e:
        logging.error(f"Error in point_in_polygon: {e}")
        return False

def load_roi_from_file(filename=None):
    if filename is None:
        filename = roi_file_path
    try:
        points = []
        with open(filename, "r") as f:
            for line in f:
                try:
                    x, y = line.strip().split(",")
                    points.append((int(x), int(y)))
                except ValueError as e:
                    logging.error(f"Invalid line in ROI file: {line.strip()}, Error: {e}")
        
        if len(points) == 4:
            logging.info(f"ROI points loaded from {filename}")
            return points
        else:
            logging.warning(f"Invalid number of points in {filename}, expected 4, got {len(points)}")
            return None
    except FileNotFoundError:
        logging.warning(f"ROI file {filename} not found")
        return None
    except Exception as e:
        logging.error(f"Could not load ROI points: {e}")
        return None

def create_mask(frame, roi_points):
    try:
        if len(roi_points) < 3:
            logging.error("Not enough ROI points to create a polygon")
            return np.ones(frame.shape[:2], dtype=np.uint8)
            
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        roi_polygon = np.array(roi_points, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [roi_polygon], 255)
        return mask
    except Exception as e:
        logging.error(f"Error creating mask: {e}")
        return np.ones(frame.shape[:2], dtype=np.uint8)

def extract_roi(frame, mask):
    try:
        roi = cv2.bitwise_and(frame, frame, mask=mask)
        return roi
    except Exception as e:
        logging.error(f"Error extracting ROI: {e}")
        return frame

def resize_frame(frame, width, height):
    """Resize the frame to the specified dimensions."""
    try:
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    except Exception as e:
        logging.error(f"Error resizing frame: {e}")
        return frame

def normalize_frame(frame):
    """Normalize pixel values to range [0, 1]."""
    try:
        return frame.astype(np.float32) / 255.0
    except Exception as e:
        logging.error(f"Error normalizing frame: {e}")
        return frame

def main():
    global roi_points, roi_selected
    
    try:
        loaded_roi = load_roi_from_file(roi_file_path)
        if loaded_roi:
            roi_points = loaded_roi
            roi_selected = True
        else:
            roi_points = default_roi_points
            logging.warning("Using default ROI points")
        

        try:
            if mode.lower() == "video":
                cap = cv2.VideoCapture(video_path)
                logging.info(f"Inferencing on video: {video_path}")
            elif mode.lower() == "rtsp":
                cap = VideoCapture(rtsp_link)
                logging.info(f"Inferencing on RTSP stream: {rtsp_link}")
            else:
                cap = cv2.VideoCapture(0)
                logging.info("Using OpenCV's VideoCapture with webcam")
                
            if mode.lower() != "rtsp" and not cap.isOpened():
                logging.error("Error opening video capture")
                return
                
        except (NameError, AttributeError, Exception) as e:
            logging.error(f"Error setting up video source: {e}")
            cap = cv2.VideoCapture(0)
            logging.info("Falling back to default webcam")
            
        try:
            try:
                model = YOLO(model_path_onnx, task='detect')
                logging.info("YOLO model loaded successfully from ONNX")
            except Exception:
                model = YOLO(model_path_pt, task='detect')
                logging.info("YOLO model loaded successfully from PT")
        except Exception as e:
            logging.error(f"Error loading YOLO model: {e}")
            return
        
        person_idle_times = {}
        last_positions = {}
        start_time = time.time()
        
        logging.info(f"Frame resizing enabled: {resize_width}x{resize_height}")
        logging.info(f"Frame normalization enabled: {normalize_frames}")
        
        while True:
            try:
                if mode.lower() == "rtsp":
                    frame = cap.read()
                    ret = frame is not None
                else:
                    ret, frame = cap.read()
                    
                if not ret:
                    logging.warning("Could not read frame. Exiting...")
                    break

                try:
                    original_frame = frame.copy()
                    
                    frame = resize_frame(frame, resize_width, resize_height)
                    
                    display_frame = frame.copy()
                    
                    if normalize_frames:
                        processed_frame = normalize_frame(frame)
                    else:
                        processed_frame = frame
                        
                except Exception as e:
                    logging.error(f"Error processing frame: {e}")
                    display_frame = frame
                    processed_frame = frame
                
                mask = create_mask(frame, roi_points)
                
                roi_frame = extract_roi(processed_frame, mask)

                if dis_lines.lower() == "y":
                    try:
                        roi_polygon = np.array(roi_points, np.int32)
                        roi_polygon = roi_polygon.reshape((-1, 1, 2))
                        cv2.polylines(display_frame, [roi_polygon], True, (255, 255, 255), 1)
                    except Exception as e:
                        logging.error(f"Error drawing ROI polygon: {e}")
                
                if normalize_frames:
                    inference_frame = (roi_frame * 255).astype(np.uint8)
                else:
                    inference_frame = roi_frame
                
                try:
                    results = model.track(inference_frame, iou=iou_threshold, imgsz=image_size, persist=persist, conf=conf_threshold, classes=classes)
                except Exception as e:
                    logging.error(f"Error during YOLO inference: {e}")
                    results = None
                
                active_ids = set()
                people_in_roi = 0
                idle_people_count = 0
                
                if results and len(results) > 0:
                    try:
                        boxes = results[0].boxes
                        
                        if hasattr(boxes, 'id') and boxes.id is not None:
                            try:
                                track_ids = boxes.id.int().cpu().tolist()
                                xyxy = boxes.xyxy.cpu().numpy()
                            except Exception as e:
                                logging.error(f"Error extracting tracking data: {e}")
                                track_ids = []
                                xyxy = []
                            
                            for track_id, box in zip(track_ids, xyxy):
                                try:
                                    x1, y1, x2, y2 = map(int, box)
                                    
                                    center_x = (x1 + x2) // 2
                                    center_y = (y1 + y2) // 2
                                    center_point = (center_x, center_y)
                                    
                                    try:
                                        if center_y < 0 or center_y >= mask.shape[0] or center_x < 0 or center_x >= mask.shape[1] or mask[center_y, center_x] == 0:
                                            continue
                                    except IndexError:
                                        logging.warning(f"Center point {center_point} out of bounds for mask shape {mask.shape}")
                                        continue
                                    
                                    people_in_roi += 1
                                    active_ids.add(track_id)
                                    
                                    try:
                                        cv2.circle(display_frame, center_point, 4, (255, 0, 255), -1)
                                    except Exception as e:
                                        logging.error(f"Error drawing center point: {e}")
                                    
                                    is_moving = True
                                    movement = 0
                                    if track_id in last_positions:
                                        try:
                                            prev_x, prev_y = last_positions[track_id]
                                            movement = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
                                            is_moving = movement > movement_threshold
                                        except Exception as e:
                                            logging.error(f"Error calculating movement: {e}")
                                    
                                    last_positions[track_id] = (center_x, center_y)
                                    
                                    try:
                                        current_time = time.time()
                                        if track_id not in person_idle_times:
                                            person_idle_times[track_id] = {
                                                'start_idle_time': current_time if not is_moving else None,
                                                'is_moving': is_moving,
                                                'total_idle_time': 0.0,
                                                'idle_status': False
                                            }
                                        else:
                                            if is_moving:
                                                person_idle_times[track_id]['start_idle_time'] = None
                                                person_idle_times[track_id]['is_moving'] = True
                                                person_idle_times[track_id]['idle_status'] = False
                                            elif not is_moving and person_idle_times[track_id]['is_moving']:
                                                person_idle_times[track_id]['start_idle_time'] = current_time
                                                person_idle_times[track_id]['is_moving'] = False
                                    except Exception as e:
                                        logging.error(f"Error updating idle times: {e}")
                                    
                                    idle_time = 0.0
                                    idle_status = False
                                    
                                    try:
                                        if not person_idle_times[track_id]['is_moving'] and person_idle_times[track_id]['start_idle_time'] is not None:
                                            idle_time = current_time - person_idle_times[track_id]['start_idle_time']
                                            idle_status = idle_time >= idle_alert_threshold
                                            person_idle_times[track_id]['idle_status'] = idle_status
                                    except Exception as e:
                                        logging.error(f"Error calculating idle status: {e}")
                                    
                                    if idle_status:
                                        idle_people_count += 1
                                    
                                    color = (0, 0, 255) if idle_status else (0, 255, 0)
                                    try:
                                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                                    except Exception as e:
                                        logging.error(f"Error drawing bounding box: {e}")
                                    
                                    status_text = f"IDLE" if idle_status else "Moving"
                                    time_text = f"Idle time: {idle_time:.1f}s"
                                    
                                    try:
                                        cv2.putText(display_frame, status_text, (x1, y1-10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                        cv2.putText(display_frame, time_text, (x1, y1-30), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                    except Exception as e:
                                        logging.error(f"Error adding status text: {e}")
                                    
                                    if idle_status:
                                        alert_text = "IDLE DETECTED!"
                                        try:
                                            cv2.putText(display_frame, alert_text, (x1, y1-50), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                        except Exception as e:
                                            logging.error(f"Error adding alert text: {e}")
                                except Exception as e:
                                    logging.error(f"Error processing detection: {e}")
                    except Exception as e:
                        logging.error(f"Error processing detection results: {e}")

                try:
                    expired_person_ids = [pid for pid in person_idle_times.keys() if pid not in active_ids]
                    for pid in expired_person_ids:
                        if pid in last_positions:
                            del last_positions[pid]
                        del person_idle_times[pid]
                except Exception as e:
                    logging.error(f"Error removing expired records: {e}")

                try:
                    current_time = time.time()
                    time_diff = current_time - start_time
                    if time_diff > 0:
                        fps = 1.0 / time_diff
                    else:
                        fps = 0
                    start_time = current_time
                except Exception as e:
                    logging.error(f"Error calculating FPS: {e}")
                    fps = 0

                try:
                    cv2.putText(display_frame, f"PREPARED BY: {author_name}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                    cv2.putText(display_frame, f"People in ROI: {people_in_roi}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                    cv2.putText(display_frame, f"Idle people: {idle_people_count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
                    cv2.putText(display_frame, f"Frame size: {resize_width}x{resize_height}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                    cv2.putText(display_frame, f"Norm: {'Yes' if normalize_frames else 'No'}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                except Exception as e:
                    logging.error(f"Error adding stats to frame: {e}")

                try:
                    cv2.imshow(project_name, display_frame)
                except Exception as e:
                    logging.error(f"Error displaying frame: {e}")
                    break

                try:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logging.info("User requested exit with 'q' key")
                        break
                except Exception as e:
                    logging.error(f"Error checking for key press: {e}")
                    
            except Exception as e:
                logging.error(f"Critical error in main loop: {e}")
                traceback.print_exc()
                continue

        try:
            logging.info("Cleaning up resources")
            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
            
    except Exception as e:
        logging.critical(f"Fatal error in main function: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Program terminated by keyboard interrupt")
    except Exception as e:
        logging.critical(f"Unhandled exception: {e}")
        traceback.print_exc()
