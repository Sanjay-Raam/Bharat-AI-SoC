# Real-Time Road Anomaly Detection

### Model Architecture: YOLOv11n (Nano)
For this project, we selected the **YOLOv11n** architecture. As the "nano" variant of the YOLOv11 family, it provides an exceptional balance between high accuracy and low computational cost, making it ideal for real-time video processing and edge-device deployment.

### Dataset:
The model was trained from fine-tuned on custom dataset consisting of over 17,000 annotated images. This large and diverse dataset ensures the model can accurately generalize and detect potholes across various lighting conditions, road types, and camera angles.

### Optimization:
To ensure smooth, real-time performance without requiring heavy GPU resources during inference, we heavily optimized the deployment pipeline:
- **INT8 Quantization (TFLite):** The trained YOLOv11n model was exported to TensorFlow Lite (`best_int8.tflite`) using 8-bit integer (INT8) quantization. This reduces the model footprint significantly and accelerates inference speeds on standard CPUs and edge devices with minimal loss in mean Average Precision (mAP).
- **Targeted Frame Sampling:** Processing every single frame of a 30fps video is computationally expensive and unnecessary for stationary objects like potholes. The script is optimized to process a `TARGET_FPS` of 5. It skips frames while maintaining an accurate map of road anomaly
- **Optimized Tensor Operations:** The pipeline normalizes frame data efficiently and manually decodes the YOLO bounding boxes, utilizing OpenCV's built-in `dnn.NMSBoxes` for highly optimized Non-Maximum Suppression.

### Tracking & Performance:
- **Centroid Tracking:** We implemented a custom, lightweight Euclidean distance tracker. By calculating the distance between the center of bounding boxes across frames (`MAX_TRACK_DISTANCE = 150 pixels`), the system can associate a detection with an existing anomaly.
- **Redundancy Prevention:** When a "new" anomaly is detected, it is assigned a unique ID and a snapshot is saved with a red bounding box highlight. If the pothole is recognized from a previous frame, the system simply updates its coordinates without saving duplicate images.
- **Memory Management:** To prevent memory bloat over long videos, the tracker includes a cleanup routine. Anomalies that leave the frame and are not seen for a set duration (`MAX_DISAPPEARED_FRAMES = 25`) are automatically purged from system memory.

### Installation:
```
sudo apt update
sudo apt install python3-opencv python3-tflite-runtime libatlas-base-dev libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev git
cd $HOME
git clone https://github.com/Sanjay-Raam/Bharat-AI-SoC.git
mv Bharat-AI-Soc Anomaly
```

### Usage:
#### Running Manually
For testing the model, input taken from Pi Camera, run the script
```
python3 $HOME/Anomaly/detect.py
```

For testing the model, input taken from an input file, use -i flag
```
python3 $HOME/Anomaly/detect.py -i /path/to/input
```

#### Running as a Background Daemon
For continuous, hands-off deployment, the script is configured to run as a background service using systemd. This ensures the script starts automatically on boot, runs invisibly in the background, and restarts automatically if it encounters an error.
```
cd $HOME/Anomaly
sed -i "s/USER/$USER/g" anomaly.service
sudo cp anomaly.service /etc/systemd/system
sudo systemctl daemon-reload
sudo systemctl enable anomaly.service
sudo systemctl start anomaly.service
```
