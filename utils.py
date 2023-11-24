import numpy as np
import cv2


def mod(a, b):
    """find a % b"""
    floored = np.floor_divide(a, b)
    return np.subtract(a, np.multiply(floored, b))

def sigmoid(x):
    """apply sigmoid actiation to numpy array"""
    return 1/ (1 + np.exp(-x))
    
def sigmoid_and_argmax2d(inputs, threshold, interpreter, output_details):
    """return y,x coordinates from heatmap"""
    #v1 is 9x9x17 heatmap
    v1 = interpreter.get_tensor(output_details[0]['index'])[0]
    height = v1.shape[0]
    width = v1.shape[1]
    depth = v1.shape[2]
    reshaped = np.reshape(v1, [height * width, depth])
    reshaped = sigmoid(reshaped)
    #apply threshold
    reshaped = (reshaped > threshold) * reshaped
    coords = np.argmax(reshaped, axis=0)
    yCoords = np.round(np.expand_dims(np.divide(coords, width), 1)) 
    xCoords = np.expand_dims(mod(coords, width), 1) 
    return np.concatenate([yCoords, xCoords], 1)

def get_offset_point(y, x, offsets, keypoint, num_key_points):
    """get offset vector from coordinate"""
    y_off = offsets[y,x, keypoint]
    x_off = offsets[y,x, keypoint+num_key_points]
    return np.array([y_off, x_off])
    

def get_offsets(output_details, coords, interpreter, num_key_points=17):
    """get offset vectors from all coordinates"""
    offsets = interpreter.get_tensor(output_details[1]['index'])[0]
    offset_vectors = np.array([]).reshape(-1,2)
    for i in range(len(coords)):
        heatmap_y = int(coords[i][0])
        heatmap_x = int(coords[i][1])
        #make sure indices aren't out of range
        if heatmap_y >8:
            heatmap_y = heatmap_y -1
        if heatmap_x > 8:
            heatmap_x = heatmap_x -1
        offset_vectors = np.vstack((offset_vectors, get_offset_point(heatmap_y, heatmap_x, offsets, i, num_key_points)))  
    return offset_vectors

def draw_lines(keypoints, image, bad_pts):
    """connect important body part keypoints with lines"""
    #color = (255, 0, 0)
    color = (0, 255, 0)
    thickness = 2
    #refernce for keypoint indexing: https://www.tensorflow.org/lite/models/pose_estimation/overview
    body_map = [[5,6], [5,7], [7,9], [5,11], [6,8], [8,10], [6,12], [11,12], [11,13], [13,15], [12,14], [14,16]]
    for map_pair in body_map:
        #print(f'Map pair {map_pair}')
        if map_pair[0] in bad_pts or map_pair[1] in bad_pts:
            continue
        start_pos = (int(keypoints[map_pair[0]][1]), int(keypoints[map_pair[0]][0]))
        end_pos = (int(keypoints[map_pair[1]][1]), int(keypoints[map_pair[1]][0]))
        image = cv2.line(image, start_pos, end_pos, color, thickness)
    return image

def is_namaskar_image(keypoint_positions, drop_pts):
	namaskar_keypoints_posns = [5,6,7,8,9,10]
	
	#print("Is namasjar started")
	for keypoint in namaskar_keypoints_posns:
		if keypoint in drop_pts:
			print("Essential point in drop points")
			return False
	
	left_shoulder_coo = keypoint_positions[5]
	right_shoulder_coo = keypoint_positions[6] 
	left_elbow_coo = keypoint_positions[7]
	right_elbow_coo = keypoint_positions[8]
	left_wrist_coo = keypoint_positions[9]
	right_wrist_coo = keypoint_positions[10]
	
	shoulder_y = max(left_shoulder_coo[0], right_shoulder_coo[0])
	elbo_y = min(left_elbow_coo[0], right_elbow_coo[0])
	#print(f"Shoulder y: {shoulder_y}")
	#print(f"Elbo y: {elbo_y}")
	
	left_side_x = min(left_shoulder_coo[1],  left_elbow_coo[1])
	right_side_x = max(right_shoulder_coo[1],  right_elbow_coo[1])
	#print(f"left_side_x: {left_side_x}")
	#print(f"right_side_x: {right_side_x}")
	
	if shoulder_y < left_wrist_coo[0] < elbo_y and shoulder_y < right_wrist_coo[0] < elbo_y:
		if right_side_x < left_wrist_coo[1] <left_side_x and right_side_x < right_wrist_coo[1] <left_side_x:
			return True
	
	return False
	
	
def is_namaskar_pose(frame_nam_status, no_frames_to_examine=6):
	curr_frame_nam_status = frame_nam_status[-no_frames_to_examine:]
	if len(curr_frame_nam_status) < no_frames_to_examine:
		return False
	
	curr_frame_nam_status_count = len([ele for ele in curr_frame_nam_status if ele==True])
	print(f"curr_frame_nam_status_count:{curr_frame_nam_status_count}")
	if curr_frame_nam_status_count > no_frames_to_examine / 2:
		return True
	else:
		return False

def is_area_clear(frame_per_present, no_frames_to_examine=6):
	curr_frames_per_present = frame_per_present[-no_frames_to_examine:]
	if len(curr_frames_per_present) < no_frames_to_examine:
		return False
	curr_frames_per_not_present_count = len([ele for ele in curr_frames_per_present if not ele])
	if curr_frames_per_not_present_count > no_frames_to_examine/ 2:
		return True
	else:
		return False
	
	
