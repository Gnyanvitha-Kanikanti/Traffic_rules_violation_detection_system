import cv2
import os
import numpy as np
import pytesseract
import re
import uuid
from inference_sdk import InferenceHTTPClient
from ultralytics import YOLO
os.chdir(r"C:\Users\gnyanvitha\Desktop\Major_Project\Major")
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
main_model = YOLO("yolov8m.pt")
plate_model = YOLO("license_plate_detector.pt")
coco = main_model.model.names
TargetLabels = ["bicycle", "car", "motorcycle", "bus", "truck", "traffic light"]
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="" # add personal API key from roboflow 
)
cap = cv2.VideoCapture("f.mp4")
ret, prev_frame = cap.read()
if not ret:
    print("Error reading video.")
    cap.release()
    exit()
prev_frame = cv2.resize(prev_frame, (1100, 700))
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
ROI = []
drawing_done = False
traffic_light_present = False
ROI_np = None
valid_plate_texts = []
plate_window = np.zeros((100, 300, 3), dtype=np.uint8)
temp_filename = "temp_rf_" + str(uuid.uuid4()) + ".jpg"
def click_roi(event, x, y, flags, param):
    global ROI, drawing_done
    if event == cv2.EVENT_LBUTTONDOWN and not drawing_done:
        ROI.append((x, y))
        if len(ROI) == 4:
            drawing_done = True
def detect_scene_change(frame1, frame2, threshold=0.6):
    h1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
    h2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
    h1 = cv2.normalize(h1, h1).flatten()
    h2 = cv2.normalize(h2, h2).flatten()
    correlation = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    return correlation < threshold
def get_traffic_light_color(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    green = cv2.inRange(hsv, (40, 70, 50), (80, 255, 255))
    if np.sum(red1 > 0) + np.sum(red2 > 0) > np.sum(green > 0):
        return "red"
    elif np.sum(green > 0) > 50:
        return "green"
    return "unknown"
def draw_text_with_background(frame, text, position, font, scale, text_color, bg_color, border_color, thickness=2, padding=5):
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    cv2.rectangle(frame, (x - padding, y - th - padding), (x + tw + padding, y + baseline + padding), bg_color, cv2.FILLED)
    cv2.rectangle(frame, (x - padding, y - th - padding), (x + tw + padding, y + baseline + padding), border_color, thickness)
    cv2.putText(frame, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1100, 700))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if detect_scene_change(prev_gray, gray):
        ROI = []
        drawing_done = False
        traffic_light_present = False
        print("Scene changed! Please select ROI (4 clicks).")
        cv2.namedWindow("Select ROI")
        cv2.setMouseCallback("Select ROI", click_roi)
        while not drawing_done:
            temp = frame.copy()
            for point in ROI:
                cv2.circle(temp, point, 5, (0, 0, 255), -1)
            cv2.imshow("Select ROI", temp)
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyWindow("Select ROI")
        ROI_np = np.array(ROI)
        results = main_model.predict(frame, conf=0.5)
        for result in results:
            for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                label = coco[int(cls)]
                if label == "traffic light":
                    traffic_light_present = True
                    break
    if ROI_np is not None:
        cv2.polylines(frame, [ROI_np], True, (255, 0, 0), 2)
    if traffic_light_present:
        traffic_light_color = "unknown"
        results = main_model.predict(frame, conf=0.5)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            for box, conf, cls in zip(boxes, confs, classes):
                label = coco[int(cls)]
                x1, y1, x2, y2 = map(int, box)
                if label == "traffic light":
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        traffic_light_color = get_traffic_light_color(crop)
                    draw_text_with_background(frame, f"Traffic Light: {traffic_light_color.upper()}",
                                              (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                              (255, 255, 255), (0, 0, 0), (0, 0, 255))
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 0, 255) if traffic_light_color == "red" else (0, 255, 0), 2)
                elif label in TargetLabels:
                    box_color = (0, 255, 0)
                    if traffic_light_color == "red":
                        pt1_in = cv2.pointPolygonTest(ROI_np, (x1, y1), False) >= 0
                        pt2_in = cv2.pointPolygonTest(ROI_np, (x2, y2), False) >= 0
                        if pt1_in or pt2_in:
                            draw_text_with_background(frame, f"{label.capitalize()} violated red light!", (10, 30),
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                                      (255, 255, 255), (0, 0, 0), (0, 0, 255))
                            box_color = (0, 0, 255)
                            plate_results = plate_model.predict(frame, conf=0.5)
                            for pres in plate_results:
                                for pbox in pres.boxes.xyxy:
                                    px1, py1, px2, py2 = map(int, pbox)
                                    plate_crop = frame[py1:py2, px1:px2]
                                    if plate_crop.size > 0:
                                        plate_text = pytesseract.image_to_string(plate_crop, config='--psm 7').strip()
                                        pattern = r'^[A-Za-z]{2}\s\d{4}$'
                                        if re.match(pattern, plate_text):
                                            if plate_text not in valid_plate_texts:
                                                valid_plate_texts.append(plate_text)
                                        cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 0), 2)
                                        break
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    draw_text_with_background(frame, f"{label}, {conf*100:.2f}%",
                                              (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                              (255, 255, 255), (0, 0, 0), box_color)
    else:
        cv2.imwrite(temp_filename, frame)
        try:
            result = CLIENT.infer(temp_filename, model_id="bike-helmet-detection-2vdjo/2")
            for pred in result["predictions"]:
                x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
                x1, y1 = x - w // 2, y - h // 2
                x2, y2 = x + w // 2, y + h // 2
                label = pred["class"]
                confidence = pred["confidence"]
                color = (0, 255, 0) if "Helmet" in label else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                draw_text_with_background(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), (0, 0, 0), color)
        except Exception as e:
            print("Helmet detection failed:", e)
    plate_window[:] = 0
    y_offset = 40
    for text in valid_plate_texts[-3:]:
        cv2.putText(plate_window, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)
        y_offset += 30
    cv2.imshow("Unified Detection", frame)
    cv2.imshow("Plate Text", plate_window)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    prev_gray = gray.copy()
cap.release()
cv2.destroyAllWindows()
if os.path.exists(temp_filename):
    os.remove(temp_filename)