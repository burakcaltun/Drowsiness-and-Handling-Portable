Real-Time Monitoring System for Drowsiness and Hand-in-Warning-Area Detection
This repository contains a real-time monitoring system that uses computer vision techniques to detect drowsiness and hands entering a warning area. The system utilizes YOLO (You Only Look Once) for hand detection and Dlib for facial landmark detection to monitor eye movements for drowsiness detection.

Prerequisites
Python 3.x
OpenCV
Pygame
Dlib
Numpy
Scipy
cvzone
Imutils
Getting Started
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/real-time-monitoring.git
cd real-time-monitoring
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Download the YOLO weights and configuration files. You can find the necessary files here.

Run the monitoring system:

bash
Copy code
python monitoring_system.py -n normal -s 416 -c 0.1 -nh 4
Usage
The system uses YOLO for hand detection. You can choose different YOLO configurations using the -n option (normal, tiny, prn, v4-tiny).

Adjust the confidence level with the -c option.

Specify the size of the hands to be detected using the -s option.

Set the number of hands to be monitored with the -nh option.

Hand Warning Area Configuration
Run the warning_area_config.py script to configure the warning area.

bash
Copy code
python warning_area_config.py
Left-click on the image to add warning areas. Right-click to remove a warning area.

Press the 'ESC' key to exit the configuration.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
YOLO (You Only Look Once) - Link
Dlib - Link
cvzone - Link
Authors
[Your Name]
Contributing
Feel free to open issues and pull requests!
