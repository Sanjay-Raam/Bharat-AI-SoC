import cv2
import numpy as np
import tensorflow as tf
import os
import math
import sys

# --- CONFIGURATION ---
MODEL_PATH = "best_int8.tflite"
VIDEO_PATH = "0"
OUTPUT_FOLDER = "Detection_Logs"

if len(sys.argv) == 3:
    if sys.argv[1] == '-i':
        VIDEO_PATH = sys.argv[2]

CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4
TARGET_FPS = 5 

# --- TRACKING SETTINGS ---
# Distance (pixels) to decide if it's the same pothole or a new one
MAX_TRACK_DISTANCE = 150  
# If we don't see a pothole ID for this many frames, forget it (so we don't match it next frame)
MAX_DISAPPEARED_FRAMES = 25 

MODEL_WIDTH = 640
MODEL_HEIGHT = 640
DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080

# --- SETUP ---
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

print("Loading TFLite Model...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(VIDEO_PATH)
video_fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = max(1, int(video_fps / TARGET_FPS))
frame_count = 0

# --- MEMORY ---
# Stores { ID : {'center': (x,y), 'missing_count': 0} }
tracked_objects = {}
next_object_id = 0

# cv2.namedWindow("Pothole Detection", cv2.WINDOW_NORMAL)
# cv2.setWindowProperty("Pothole Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_interval != 0:
        continue

    # --- 1. INFERENCE ---
    model_frame = cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT))
    input_data = cv2.cvtColor(model_frame, cv2.COLOR_BGR2RGB)
    input_data = input_data.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # --- 2. DECODING ---
    predictions = np.squeeze(output_data).T
    class_scores = predictions[:, 4:] 
    max_scores = np.max(class_scores, axis=1)
    
    valid_indices = np.where(max_scores > CONFIDENCE_THRESHOLD)[0]
    valid_predictions = predictions[valid_indices]
    valid_scores = max_scores[valid_indices]

    boxes = []
    centers = []
    confidences = []

    x_scale = DISPLAY_WIDTH / MODEL_WIDTH
    y_scale = DISPLAY_HEIGHT / MODEL_HEIGHT

    for i, pred in enumerate(valid_predictions):
        x_norm, y_norm, w_norm, h_norm = pred[:4]
        
        x = x_norm * MODEL_WIDTH * x_scale
        y = y_norm * MODEL_HEIGHT * y_scale
        w = w_norm * MODEL_WIDTH * x_scale
        h = h_norm * MODEL_HEIGHT * y_scale
        
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        
        boxes.append([x1, y1, int(w), int(h)])
        centers.append((x, y))
        confidences.append(float(valid_scores[i]))

    # --- 3. NMS ---
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    
    display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    # Get final list of detections for this frame
    current_frame_centers = []
    current_frame_boxes = []
    
    if len(indices) > 0:
        for i in indices.flatten():
            # EXTRACT DATA
            box = boxes[i]
            center = centers[i]
            confidence = confidences[i]
            
            current_frame_centers.append(center)
            current_frame_boxes.append(box)

            # --- VISUALIZATION (ALWAYS DRAW) ---
            # This runs regardless of the tracker. You will see every detection.
            x, y, w, h = box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            label = f"{int(confidence * 100)}%"
            cv2.putText(display_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # --- 4. TRACKING & SAVING LOGIC (Background Process) ---
    
    # Mark existing objects as potentially missing (we will un-mark them if we find a match)
    for obj_id in tracked_objects:
        tracked_objects[obj_id]['missing_count'] += 1

    for i, center in enumerate(current_frame_centers):
        cx, cy = center
        matched_id = None
        min_dist = 99999.0

        # Try to match this detection to an existing ID
        for obj_id, data in tracked_objects.items():
            saved_center = data['center']
            dist = math.sqrt((cx - saved_center[0])**2 + (cy - saved_center[1])**2)
            
            if dist < MAX_TRACK_DISTANCE and dist < min_dist:
                min_dist = dist
                matched_id = obj_id

        if matched_id is not None:
            # IT IS AN OLD ANOMALY -> Update position, Reset missing count, DON'T SAVE
            tracked_objects[matched_id]['center'] = center
            tracked_objects[matched_id]['missing_count'] = 0
            
            # Optional: Draw ID so you know it's being tracked
            bx, by, bw, bh = current_frame_boxes[i]
            cv2.putText(display_frame, f"ID:{matched_id}", (bx, by+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
        else:
            # IT IS A NEW ANOMALY -> Save Image, Create New ID
            filename = f"{OUTPUT_FOLDER}/anomaly_ID{next_object_id}_frame{frame_count}.jpg"
            
            # Save the frame
            save_img = display_frame.copy() 
            # (Optional) If you want the saved image to have the RED box showing exactly which one triggered it:
            bx, by, bw, bh = current_frame_boxes[i]
            cv2.rectangle(save_img, (bx, by), (bx+bw, by+bh), (0, 0, 255), 3)
            
            cv2.imwrite(filename, save_img)
            print(f"saved NEW Anomaly ID: {next_object_id}")

            # Register in memory
            tracked_objects[next_object_id] = {'center': center, 'missing_count': 0}
            next_object_id += 1

    # --- 5. CLEANUP OLD IDS ---
    # Remove objects that haven't been seen for too long
    ids_to_delete = []
    for obj_id, data in tracked_objects.items():
        if data['missing_count'] > MAX_DISAPPEARED_FRAMES:
            ids_to_delete.append(obj_id)
            
    for obj_id in ids_to_delete:
        del tracked_objects[obj_id]

    #cv2.imshow("Anomaly Detection", display_frame)

    #if cv2.waitKey(200) & 0xFF == ord('q'):
    #    break

cap.release()
cv2.destroyAllWindows()
