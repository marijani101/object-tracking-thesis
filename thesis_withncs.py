#python may27.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/dashcam_boston.mp4 \
#	--output output/output_01.avi
# python thesis withncs.py --graph
#works, does only detection on the first frame. was updated on apr30. added q to quit

from imutils.video import VideoStream
from imutils.video import FPS
from collections import OrderedDict
import argparse
import imutils
import time
import cv2
import numpy as np
#import pysnooper

#def supertracker(boxrange)
	

#@pysnooper.snoop()

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]


# frame dimensions should be sqaure
PREPROCESS_DIMS = (300, 300)
DISPLAY_DIMS = (900, 900)
 
# calculate the multiplier needed to scale the bounding boxes
DISP_MULTIPLIER = DISPLAY_DIMS[0] // PREPROCESS_DIMS[0]

def preprocess_image(input_image):
	# preprocess the image
	preprocessed = cv2.resize(input_image, PREPROCESS_DIMS)
	preprocessed = preprocessed - 127.5
	preprocessed = preprocessed * 0.007843
	preprocessed = preprocessed.astype(np.float16)
 
	# return the image to the calling function
	return preprocessed

def predict(image, graph):
	# preprocess the image
	image = preprocess_image(image)
 
	# send the image to the NCS and run a forward pass to grab the
	# network predictions
	graph.LoadTensor(image, None)
	(output, _) = graph.GetResult()
 
	# grab the number of valid object predictions from the output,
	# then initialize the list of predictions
	num_valid_boxes = output[0]
	predictions = []
	# loop over results
	for box_index in range(num_valid_boxes):
		# calculate the base index into our array so we can extract
		# bounding box information
		base_index = 7 + box_index * 7
 
		# boxes with non-finite (inf, nan, etc) numbers must be ignored
		if (not np.isfinite(output[base_index]) or
			not np.isfinite(output[base_index + 1]) or
			not np.isfinite(output[base_index + 2]) or
			not np.isfinite(output[base_index + 3]) or
			not np.isfinite(output[base_index + 4]) or
			not np.isfinite(output[base_index + 5]) or
			not np.isfinite(output[base_index + 6])):
			continue
 
		# extract the image width and height and clip the boxes to the
		# image size in case network returns boxes outside of the image
		# boundaries
		(h, w) = image.shape[:2]
		x1 = max(0, int(output[base_index + 3] * w))
		y1 = max(0, int(output[base_index + 4] * h))
		x2 = min(w,	int(output[base_index + 5] * w))
		y2 = min(h,	int(output[base_index + 6] * h))
 
		# grab the prediction class label, confidence (i.e., probability),
		# and bounding box (x, y)-coordinates
		pred_class = int(output[base_index + 1])
		pred_conf = output[base_index + 2]
		pred_boxpts = ((x1, y1), (x2, y2))
 
		# create prediciton tuple and append the prediction to the
		# predictions list
		prediction = (pred_class, pred_conf, pred_boxpts)
		predictions.append(prediction)
 
	# return the list of predictions to the calling function
	return predictions
# load our serialized model from disk
#print("[INFO] loading model...")
#net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--prototxt", required=True,
# 	help="path to Caffe 'deploy' prototxt file")
# ap.add_argument("-m", "--model", required=True,
# 	help="path to Caffe pre-trained model")
ap.add_argument("-g", "--graph", required=True,
	help="path to input graph file")
ap.add_argument("-i", "--input", type=str,
	help="path to optional input video file")
# ap.add_argument("-o", "--output", type=str,
# 	help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=40,
	help="# of skip frames between detections")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
ap.add_argument("-v", "--threshold-val", type=str, default= "10",
	help="Threshold value for bounding boxes")
args = vars(ap.parse_args())

# extract the OpenCV version info
(major, minor) = cv2.__version__.split(".")[:2]

# if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
# function to create our object tracker
if int(major) == 3 and int(minor) < 3:
	tracker = cv2.Tracker_create(args["tracker"].upper())


else:
	# initialize a dictionary that maps strings to their corresponding
	# OpenCV object tracker implementations
	OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"boosting": cv2.TrackerBoosting_create,
		"mil": cv2.TrackerMIL_create,
		"tld": cv2.TrackerTLD_create,
		"medianflow": cv2.TrackerMedianFlow_create,
		"mosse": cv2.TrackerMOSSE_create
	}

	# grab the appropriate object tracker using our dictionary of
	# OpenCV object tracker objects
	tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
	multiTracker = cv2.MultiTracker_create()

print("[INFO] finding NCS devices...")
devices = mvnc.EnumerateDevices()
 
# if no devices found, exit the script
if len(devices) == 0:
	print("[INFO] No devices found. Please plug in a NCS")
	quit()

print("[INFO] found {} devices. device0 will be used. "
	"opening device0...".format(len(devices)))
device = mvnc.Device(devices[0])
device.OpenDevice()
 

# open the CNN graph file
print("[INFO] loading the graph file into RPi memory...")
with open(args["graph"], mode="rb") as f:
	graph_in_memory = f.read()
# load the graph into the NCS
print("[INFO] allocating the graph on the NCS...")
graph = device.AllocateGraph(graph_in_memory)

# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(usePiCamera=True).start()
	time.sleep(2.0)
	inputval = 0

# otherwise, grab a reference to the video file
else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture(args["input"])
	inputval = 1

# # initialize the video writer (we'll instantiate later if need be)
# writer = None



# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None
initBB = None
bboxes = []

#initialize the fps collector for statistical purposes
a = time.strftime("%d_%b_%H_%M")
name = a+"_"+args["tracker"]+".txt"
fpsfile = open(name,"a")


########perform first detection########
#initialize total frames, to see how many frames we need in between detection and tracking.
totalFrames = 0
found = 0
skipframes = 30
currentsizeoftracker = 0
sizes = 0
newbboxes = []
thresholdval = 10
fps = FPS().start()

#loops until initial detection is found. 
while True:
	try:
		frame = vs.read()
		frame = frame[1] if args.get("input", False) else frame

		image_for_result = frame.copy()
		image_for_result = cv2.resize(image_for_result, DISPLAY_DIMS)
	 
		# use the NCS to acquire predictions
		predictions = predict(frame, graph)
		
	

		# loop over our predictions
		for (i, pred) in enumerate(predictions):
			# extract prediction data for readability
			(pred_class, pred_conf, pred_boxpts) = pred
		

		# filter out weak detections by ensuring the `confidence`
		# is greater than the minimum confidence
		if pred_conf > args["confidence"] && CLASSES[idx] == "car":
			# print prediction to terminal
			print("[INFO] Prediction #{}: class={}, confidence={}, "
				"boxpoints={}".format(i, CLASSES[pred_class], pred_conf,
				pred_boxpts))
			(ptA, ptB) = (pred_boxpts[0], pred_boxpts[1])
			ptA = (ptA[0] * DISP_MULTIPLIER, ptA[1] * DISP_MULTIPLIER)
			ptB = (ptB[0] * DISP_MULTIPLIER, ptB[1] * DISP_MULTIPLIER)
			(startX, startY) = (ptA[0], ptA[1])
			y = startY - 15 if startY - 15 > 15 else startY + 15

			cv2.rectangle(image_for_result, ptA, ptB, (255, 153, 0), 2)
			print("ptA")
			print(ptA)
			print("ptB")
			print(ptB)	

			cv2.imshow("Output", image_for_result)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break

			fps.update()
		except KeyboardInterrupt:
		break
 		
	# if there's a problem reading a frame, break gracefully
		except AttributeError:
			break
fps.stop()
 
# destroy all windows if we are displaying them

cv2.destroyAllWindows()
 
# stop the video stream
vs.stop()
 
# clean up the graph and device
graph.DeallocateGraph()
device.CloseDevice()
 
# display FPS information
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
		
