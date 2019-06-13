#python may27.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/dashcam_boston.mp4 \
#	--output output/output_01.avi

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
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str,
	help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
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

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
	inputval = 0

# otherwise, grab a reference to the video file
else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture(args["input"])
	inputval = 1

# initialize the video writer (we'll instantiate later if need be)
writer = None



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
while found == 0:
	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame

	# check to see if we have reached the end of the stream
	#if frame is None:
	#	break

	# resize the frame (so we can process it faster) and grab the
	# frame dimensions
	frame = imutils.resize(frame, width=500)
	(H, W) = frame.shape[:2]
		
	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

	
	blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
	net.setInput(blob)
	detections = net.forward()

	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated
		# with the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by requiring a minimum
		# confidence
		if confidence > args["confidence"]:
			# extract the index of the class label from the
			# detections list
			idx = int(detections[0, 0, i, 1]) 

			#if CLASSES[idx]
			# if the class label is not a car, ignore it
			if CLASSES[idx] == "car":
				
				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				(startX, startY, endX, endY) = box.astype("int")
				#print(startX, startY, endX, endY)
				width = endX-startX
				height = endY-startY
				initBB = (startX,startY,width,height)	
				#print(initBB)
				bboxes.append(initBB)
				found = 1
				totalFrames +=1

				#tracker.init(frame, initBB)
				#(x, y, w, h) = [int(v) for v in box]
				#cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)
				#fps = FPS().start()
				#totalFrames+=1
				#found = 1

for bbox in bboxes:
	multiTracker.add(cv2.TrackerCSRT_create(),frame,bbox)
	retval	=multiTracker.getObjects()
	#print(type(bbox))
	#print(retval)
	#(x, y, w, h) = [int(v) for v in bbox]
	#cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)

cv2.imshow("MultiTracker", frame)
fps.update()
anobox = []
dellist = []  
while True:
	
		
	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame

	# check to see if we have reached the end of the stream
	if frame is None:
		break

	# resize the frame (so we can process it faster) and grab the
	# frame dimensions
	frame = imutils.resize(frame, width=500)
	(H, W) = frame.shape[:2]




	if totalFrames %skipframes ==0:

		newbboxes = []
		#print("re-detection after 30 secs") ##debuging purposes
		blober = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
		net.setInput(blober)
		detections = net.forward()

		for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated
			# with the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by requiring a minimum
			# confidence
			if confidence > args["confidence"]:
				# extract the index of the class label from the
				# detections list
				idx = int(detections[0, 0, i, 1]) 

				#if CLASSES[idx]
				# if the class label is not a car, ignore it
				if CLASSES[idx] == "car":
					sizes+=1
					box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
					(startX, startY, endX, endY) = box.astype("int")
					#print(startX, startY, endX, endY)
					width = endX-startX
					height = endY-startY
					initBB = (startX,startY,width,height)	
					#print(sizes)
					newbboxes.append(initBB)

		for lilbox in retval:
			(x1,y1,w1,h1) = [int(v) for v in bbox]
			for newbox in newbboxes:
				(x2,y2,w2,h2) = [int(v) for v in newbox]	
				if ((abs(x1-x2) < thresholdval)):
					a = newbboxes.index(newbox)	
					dellist.append(a)
				elif((abs(y1-y2) < thresholdval)):
					a = newbboxes.index(newbox)	
					dellist.append(a)
				
		#print(dellist)
		#print(newbboxes)
		dellist =list(OrderedDict.fromkeys(dellist))
		a = dellist[::-1]
		#print(a)
		try:
			if len(newbboxes) < len(dellist):
				print(newbboxes)
				newbboxes =[]
			
			for index_i in a:
				if index_i <=len(newbboxes):
					del newbboxes[index_i] 
				else:
					continue
		except IndexError:
			pass


		

		#2 #####################################
		if len(newbboxes)>0: 
			for nbbox in newbboxes:
				multiTracker.add(cv2.TrackerCSRT_create(),frame,nbbox)

		
		#bboxes = []




	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

	(success, box1) = multiTracker.update(frame)
	retval	=multiTracker.getObjects()	
	#print(type(retval))
	#print(retval)
	currentsizeoftracker = len(retval)
	#print(len(retval)) #for debuging purposes

	if success:
		for bbox in box1:
			#print(bbox)
			(x, y, w, h) = [int(v) for v in bbox]
			if x >0 and y >0:  
				cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)
	
	info = [
			("Tracker", args["tracker"]),
			("Success", "Yes" if success else "No"),
			#("FPS", "{:.2f}".format(fps.fps())),
	]

	# loop over the info tuples and draw them on our frame
	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	# show the output frame
	cv2.imshow("MultiTracker", frame)
	key = cv2.waitKey(1) & 0xFF
	fps.update()
	totalFrames+=1
	
	#press q to quit
	if key == ord("q"):
		break
		
	##increasing confidence## with extra detection
	#30 is the number of skip-frames
	########################################
	
fps.stop() ##added line
print("[INFO] elasped time: {:.4f}".format(fps.elapsed())) ##added line
print("[INFO] approx. FPS: {:.4f}".format(fps.fps()))##Added line
fpsfile.write("Marijani Hussein \n \n")
fpsfile.write("tracker type: " +args["tracker"]+"\n")
fpsfile.write("time elapsed{:.4f} \n".format(fps.elapsed()))
fpsfile.write("FPS {:.4f} \n".format(fps.fps()))

# if we are using a webcam, release the pointer
if inputval == 0:
	vs.stop
#if not args.get("video", False):
#	vs.stop()

# otherwise, release the file pointer
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()


