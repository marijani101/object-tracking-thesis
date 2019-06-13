#!/bin/bash
echo "Marijani Karanda"
echo "testing script for thesis"
echo "############################"
echo "  "
cd 
echo "opencv environment"
cd ~/thesis
echo "kcf tracker"
python thesis.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/dashcam_boston.mp4 -t kcf 
echo "csrt tracker"
python thesis.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/dashcam_boston.mp4 -t csrt 
echo "boosting tracker"
python thesis.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/dashcam_boston.mp4 -t boosting 
echo "medianflow tracker"
python thesis.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/dashcam_boston.mp4 -t medianflow 
echo "mil tracker"
python thesis.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/dashcam_boston.mp4 -t mil 
echo "tld tracker"
python thesis.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/dashcam_boston.mp4 -t tld
echo "mosse tracker"
python thesis.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/dashcam_boston.mp4 -t mosse 
