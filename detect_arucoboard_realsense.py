import cv2
import numpy as np
import pyrealsense2 as rs
import os
import argparse
import json
import utils
import shutil

###############################
# load the aruco board config #
###############################
parser = argparse.ArgumentParser(description='Aruco Board Generator.')
parser.add_argument('--config',
					type=str,
					default='sl200_ml120',
					help='config json, located in configs/aruco/')
parser.add_argument('--name',
					type=str,
					default='empty',
					help='name of saved dataset, located in out/datasets/')
args = parser.parse_args()
cfg_name = args.config
if cfg_name[-5:] != '.json':
	cfg_name = cfg_name + '.json'
cfg_pth = os.path.join(os.getcwd(), 'configs', 'aruco', cfg_name)
with open(cfg_pth, 'r') as f:
	cfg = json.load(f)

sl = cfg['sl']
ml = cfg['ml']
ids = cfg['ids']
charucodict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard((3,3),sl,ml,charucodict,np.array([ids]))
origin_offset = np.array([[-300,-300,0]])
Rx180 = np.array([[ 1, 0,  0],
				  [ 0 ,-1, 0],
				  [ 0 , 0, -1]])

#########################
# load realsense config #
#########################
pipe = rs.pipeline()
rscfg = rs.config()
width, height = 1280, 800
rscfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
profile = pipe.start(rscfg)

################################
# getting realsense intrinsics #
################################
intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
K_full = np.array([[intrin.fx, 0, intrin.ppx],
				   [0, intrin.fy, intrin.ppy],
				   [0,         0,          1]])
#dist_coeffs = np.array(intrin.coeffs)

##################
# save directory #
##################
dataset_name = args.name
# check if same name done
dataset_pth = os.path.join(os.getcwd(), 'out', 'datasets', dataset_name)
if os.path.exists(dataset_pth):
	shutil.rmtree(dataset_pth, ignore_errors=False, onerror=None)
os.makedirs(dataset_pth)
os.makedirs(os.path.join(dataset_pth, 'color'))
os.makedirs(os.path.join(dataset_pth, 'intrin_ba'))
os.makedirs(os.path.join(dataset_pth, 'poses_ba'))

idx = 0
while True:
	# stream frame
	frameset = pipe.wait_for_frames()
	color_frame = frameset.get_color_frame()
	bgr = np.asanyarray(color_frame.get_data())
	
	# process the frame
	bbox = np.array([240,0,1040,800])
	bgr_crop, K_crop = utils.crop_img_by_bbox(bgr, bbox, K = K_full, crop_size = 512)
	markerCorners, markerIds, _ = cv2.aruco.detectMarkers(bgr_crop, charucodict)
	cv2.aruco.drawDetectedMarkers(bgr_crop, markerCorners, markerIds)
	
	rvec, tvec = None, None
	if markerIds is not None:
		objPoints, imgPoints = board.matchImagePoints(markerCorners, markerIds)
		objPoints = objPoints + origin_offset
		_, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, K_crop, distCoeffs=None)
		rvec = cv2.Rodrigues(cv2.Rodrigues(rvec)[0] @ Rx180)[0]
		
		cv2.drawFrameAxes(bgr_crop, K_crop, None, rvec, tvec, 100)
	# show the frame
	cv2.imshow('frame_crop',bgr_crop)
	key = cv2.waitKey(1)
	if key == ord('q'):
		break
	elif key == 32:
		if markerIds is not None and rvec is not None and tvec is not None:
			print('saving data, iter =', idx)
			bgr_crop_clean, K_crop_clean = utils.crop_img_by_bbox(bgr, bbox, K=K_full, crop_size = 512)
			# store image
			cv2.imwrite(os.path.join(dataset_pth, 'color', str(idx)+'.png'), bgr_crop_clean)
			# store intrinsic
			np.savetxt(os.path.join(dataset_pth, 'intrin_ba', str(idx)+'.txt'), K_crop_clean)
			# store poses
			pose = np.zeros((4,4))
			pose[3,3] = 1
			pose[:3,3] = np.squeeze(tvec)
			pose[:3,:3] = cv2.Rodrigues(rvec)[0]
			np.savetxt(os.path.join(dataset_pth, 'poses_ba', str(idx)+'.txt'),pose)
			# update index
			idx += 1
		else:
			print('markers not detected, data not captured!')
pipe.stop()
