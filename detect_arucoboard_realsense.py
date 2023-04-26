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
axis_offset = -int(sl * 3/2)
origin_offset = np.array([[axis_offset,axis_offset,0]])
Rx180 = np.array([[ 1, 0, 0],
				  [ 0,-1, 0],
				  [ 0, 0,-1]])

x, y, z, c = 100, 100, 200, 0
bbox3d = None

#########################
# load realsense config #
#########################
pipe = rs.pipeline()
rscfg = rs.config()
width, height = 1280, 720
rscfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
profile = pipe.start(rscfg)
bbox = np.array([int(width/2-height/2), 0, width - int(width/2-height/2), height])

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
os.makedirs(os.path.join(dataset_pth, 'reproj_box'))

idx = 0
while True:
	bbox3d = np.array([[ x, y,  c],
					   [ x, y,z+c],
					   [ x,-y,z+c],
					   [ x,-y,  c],
					   [-x, y,  c],
					   [-x, y,z+c],
					   [-x,-y,z+c],
					   [-x,-y,  c]])
	# stream frame
	frameset = pipe.wait_for_frames()
	color_frame = frameset.get_color_frame()
	bgr = np.asanyarray(color_frame.get_data())
	
	# process the frame
	bgr_crop, K_crop = utils.crop_img_by_bbox(bgr, bbox, K = K_full, crop_size = 512)
	markerCorners, markerIds, _ = cv2.aruco.detectMarkers(bgr_crop, charucodict)
	cv2.aruco.drawDetectedMarkers(bgr_crop, markerCorners, markerIds)
	
	rvec, tvec = None, None
	if markerIds is not None:
		objPoints, imgPoints = board.matchImagePoints(markerCorners, markerIds)
		objPoints = objPoints + origin_offset
		_, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, K_crop, distCoeffs=None)
		
		rmat = cv2.Rodrigues(rvec)[0] @ Rx180
		rvec = cv2.Rodrigues(rmat)[0]
		
		pose = np.eye(4)
		pose[:3,:3] = rmat
		pose[:3,3] = tvec[:,0]
		
		bbox2d = utils.reproj(K_crop, pose, bbox3d)
		utils.draw_3d_box(bgr_crop, bbox2d)
		
		cv2.drawFrameAxes(bgr_crop, K_crop, None, rvec, tvec, 100)
	# show the frame
	cv2.imshow('frame_crop',bgr_crop)
	key = cv2.waitKey(1)
	if key == ord('q'):
		break
	elif key == ord('w'):
		origin_offset[0,2] += 10
		c -= 10
		print('pose height changed to', origin_offset[0,2])
	elif key == ord('s'):
		origin_offset[0,2] -= 10
		c += 10
		print('pose height changed to', origin_offset[0,2])
	elif key == ord('e'):
		x += 10
		print('bbox x-width changed to', 2*x)
	elif key == ord('d'):
		x -= 10
		print('bbox x-width changed to', 2*x)
	elif key == ord('r'):
		y += 10
		print('bbox y-width changed to', 2*y)
	elif key == ord('f'):
		y -= 10
		print('bbox y-width changed to', 2*y)
	elif key == ord('t'):
		z += 10
		print('bbox height changed to', z)
	elif key == ord('g'):
		z -= 10
		print('bbox height changed to', z)
	elif key == 32:
		if markerIds is not None and rvec is not None and tvec is not None:
			print('saving data, iter =', idx)
			bgr_crop_clean, K_crop_clean = utils.crop_img_by_bbox(bgr, bbox, K=K_full, crop_size = 512)
			# store image
			cv2.imwrite(os.path.join(dataset_pth, 'color', str(idx)+'.png'), bgr_crop_clean)
			# store intrinsic
			np.savetxt(os.path.join(dataset_pth, 'intrin_ba', str(idx)+'.txt'), K_crop_clean)
			# store poses
			np.savetxt(os.path.join(dataset_pth, 'poses_ba', str(idx)+'.txt'), pose)
			# store 2d bounding box coordinates
			np.savetxt(os.path.join(dataset_pth, 'reproj_box', str(idx)+'.txt'), bbox2d)
			# update index
			idx += 1
		else:
			print('markers not detected, data not captured!')
pipe.stop()
np.savetxt(os.path.join(dataset_pth, 'box3d_corners.txt'), bbox3d)
