import cv2
from tflite_runtime.interpreter import Interpreter
import os
import numpy as np
import cv2
import time
from utils import sigmoid_and_argmax2d, get_offsets, draw_lines, is_namaskar_image, is_namaskar_pose, is_area_clear
import pigpio
import time


# connect to the 
pi = pigpio.pi()

#configurations
MODEL_NAME="/home/ravi/project_bappa/model/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"
imW,imH = 640, 480
outdir="/home/ravi/project_bappa/images/"

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = MODEL_NAME
min_conf_threshold = 0.5
resW, resH = 640, 480
imW, imH = int(resW), int(resH)

interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()


# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
#set stride to 32 based on model size
output_stride = 32

floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5


cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
	rval, frame = vc.read()
else:
	rval = False


frame_nam_status = []
frame_per_present = []
dispence_flag = True
while rval:
	#cv2.imshow("preview", frame)
	rval, frame = vc.read()
	key = cv2.waitKey(1)
    
    
    # Grab frame from video stream
	_, frame1 = vc.read()
	# Acquire frame and resize to expected shape [1xHxWx3]
	frame = frame1.copy()
	frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame_resized = cv2.resize(frame_rgb, (width, height))
	input_data = np.expand_dims(frame_resized, axis=0)
	frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
	
	
	# Normalize pixel values if using a floating model (i.e. if model is non-quantized)
	if floating_model:
		input_data = (np.float32(input_data) - input_mean) / input_std

	# Perform the actual detection by running the model with the image as input
	interpreter.set_tensor(input_details[0]['index'],input_data)
	interpreter.invoke() 
	  
	#get y,x positions from heatmap
	coords = sigmoid_and_argmax2d(output_details, min_conf_threshold, interpreter, output_details)
	#keep track of keypoints that don't meet threshold
	
	drop_pts = list(np.unique(np.where(coords ==0)[0]))
	#get offets from postions
	offset_vectors = get_offsets(output_details, coords, interpreter)
	#use stide to get coordinates in image coordinates
	keypoint_positions = coords * output_stride + offset_vectors
	
	#print(len(keypoint_positions))
	#print(keypoint_positions)
	#print("-"*25)
	#print(drop_pts)
	# Loop over all detections and draw detection box if confidence is above minimum threshold
	for i in range(len(keypoint_positions)):
		#don't draw low confidence points
		if i in drop_pts: 
			continue
		#if not (i == 9  or  i == 10):
		#	continue
			
		# Center coordinates
		x = int(keypoint_positions[i][1])
		y = int(keypoint_positions[i][0])
		center_coordinates = (x, y)
		radius = 2
		color = (0, 255, 0)
		thickness = 2
		cv2.circle(frame_resized, center_coordinates, radius, color, thickness)
		#if debug:
		#	cv2.putText(frame_resized, str(i), (x-4, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1) # Draw label text

	frame_resized = draw_lines(keypoint_positions, frame_resized, drop_pts)
	frame_resized_flipped = np.flip(frame_resized, axis=1)
	cv2.imshow("preview", frame_resized_flipped)
	
	#print(f"the number of drop points are {len(drop_pts)}")
	if len(drop_pts) < 17:
		frame_per_present.append(True)
	else:
		frame_per_present.append(False)
		
	if is_namaskar_image(keypoint_positions, drop_pts):
		#timestr = time.strftime("%Y%m%d-%H%M%S")
		#filename = "image_"+timestr+".jpeg"
		#filepath = os.path.join(outdir, filename)
		#cv2.imwrite(filepath, frame_resized)
		print("Namaskar in image detected")
		frame_nam_status.append(True)
	else:
		frame_nam_status.append(False)
		#print("Namaskar in image NOT detected")
		
	
	if is_namaskar_pose(frame_nam_status) and dispence_flag:
		print("Consistent Namaskar pose detected")
		
		print("Dispencing the Prasad, please collect")
		pi.set_servo_pulsewidth(12, 0)    # off
		time.sleep(1)
		pi.set_servo_pulsewidth(12, 1500) # position anti-clockwise
		time.sleep(1)
		pi.set_servo_pulsewidth(12, 1400) # middle
		time.sleep(0.5)
		pi.set_servo_pulsewidth(12, 1500) # position anti-clockwise
		time.sleep(1)
		pi.set_servo_pulsewidth(12, 0)    # off
		time.sleep(1)
		dispence_flag = False
	
	if is_namaskar_pose(frame_nam_status) and not dispence_flag:
		print("You have collected the Prasad, please let next persome come")
	
	
	if is_area_clear(frame_per_present):
		print("New person may come")
		dispence_flag = True

	if key == 27: # exit on ESC
		break

	


#cv2.imwrite("/home/ravi/project_bappa/test.jpeg", frame)
cv2.destroyWindow("preview")
vc.release()
