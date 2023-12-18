Smart Monitoring System
=======================

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Overview
--------

This repository contains a Smart Monitoring System that uses computer vision techniques to detect drowsiness and hand positions in real-time. It integrates YOLO (You Only Look Once) for object detection and Dlib for facial landmark detection to achieve these functionalities.

Features
--------

*   Drowsiness detection based on eye aspect ratio.
*   Hand position detection using YOLO.
*   Real-time video monitoring.
*   Warning area visualization and alert system.

Prerequisites
-------------

*   Python 3.x
*   OpenCV
*   Dlib
*   Pygame
*   Cvzone

Installation
------------

1.  Clone the repository:
    
    bashCopy code
    
    `git clone https://github.com/your-username/smart-monitoring-system.git cd smart-monitoring-system`
    
2.  Install the required dependencies:
    
    bashCopy code
    
    `pip install -r requirements.txt`
    

Usage
-----

1.  Run the main application:
    
    bashCopy code
    
    `python main.py -n normal -s 416 -c 0.1 -nh 4`
    
    Replace the arguments (`-n`, `-s`, `-c`, `-nh`) with your preferred configuration.
    
2.  Follow the on-screen instructions to set up the warning areas by running the `warning_area_setup.py` script.
    

Contributing
------------

1.  Fork the repository.
2.  Create a new branch: `git checkout -b feature/new-feature`.
3.  Make your changes and commit them: `git commit -am 'Add new feature'`.
4.  Push to the branch: `git push origin feature/new-feature`.
5.  Submit a pull request.

License
-------

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
