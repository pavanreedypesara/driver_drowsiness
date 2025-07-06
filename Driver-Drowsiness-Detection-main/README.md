# Driver Drowsiness Monitoring System

## Overview
Drowsy driving is a leading cause of road accidents and fatalities. This project aims to detect driver drowsiness using **visual behavior analysis** and **machine learning (SVM - Support Vector Machine)**. The system utilizes a webcam to monitor the driver’s face and detects **eye blinks** and **yawning** as indicators of drowsiness. If the system detects drowsiness, it alerts the driver.

## Features
- **Real-time Video Monitoring:** Uses a webcam to capture video frames.
- **Face and Facial Landmark Detection:** Identifies facial features such as eyes and mouth.
- **Drowsiness Detection:** Detects prolonged eye closure (20 consecutive frames) or yawning.
- **Alert System:** Displays warning messages when drowsiness is detected.

## Technologies Used
- **OpenCV:** Image processing and face detection.
- **SVM (Support Vector Machine):** Pre-trained model for drowsiness detection.
- **Python:** Programming language for implementation.
- **NumPy:** For numerical operations.

## Implementation Details

### 1. Video Capture
- Uses OpenCV’s `VideoCapture()` to access the webcam.
- Extracts video frames for analysis.

### 2. Frame Processing
- Converts each frame into a **2D array**.
- Converts images to grayscale to improve processing efficiency.

### 3. Face Detection & Facial Landmark Extraction
- Detects the driver’s face in each frame using **SVM-based** face detection.
- Extracts facial landmarks (eyes, mouth) using **pre-trained models**.

### 4. Drowsiness Detection
- Computes the **Euclidean Distance** between key facial landmarks.
- Detects:
  - **Eye blinks**: If the eyes are closed for 20 consecutive frames.
  - **Yawning**: Based on mouth opening distance.
- If either condition is met, the system triggers an **alert message**.

## How to Run the Project
1. Ensure all dependencies are installed:
   ```bash
   pip install opencv-python numpy
   ```
2. Run the application by executing:
   ```bash
   run.bat
   ```
3. Click on **Start Behaviour Monitoring Using Webcam** to begin detection.
4. If drowsiness is detected, an alert message will appear.

## Screenshots
### Webcam Streaming
![Webcam Streaming](![Screenshot (74)](https://github.com/user-attachments/assets/544f5cac-fd1c-4cb6-b5c1-02498591c229)
.png)

### Drowsiness Detected
![Drowsiness Alert](![Screenshot (79)](https://github.com/user-attachments/assets/4d9c04b8-984f-4893-8118-7dd4e73c3e08)
.png)

## Support Vector Machine (SVM) Overview
SVM is a supervised machine learning algorithm used for classification tasks. It works by creating a **hyperplane** that best separates different classes in the dataset. The **Radial Basis Function (RBF) kernel** is used in this project to improve classification accuracy.

## Conclusion
This **low-cost, real-time driver drowsiness detection system** is designed to improve road safety. It offers an effective method for detecting driver fatigue using computer vision and machine learning techniques.

## Future Enhancements
- Implement an **audio alert system**.
- Optimize the SVM model for better accuracy.
- Develop a **mobile-friendly version** for real-time deployment.

## Author
Developed by A.Revanth

