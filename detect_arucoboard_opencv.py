import cv2
import numpy as np
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
parser.add_argument('--intrin',
					type=str,
					default='intrin_ps3',
					help='3x3 numpy array saved as a .txt, located in configs/intrin/')
parser.add_argument('-wd', '--width',
					type=int,
					default=640,
					help='width of image from camera.')
parser.add_argument('-ht', '--height',
					type=int,
					default=480,
					help='height of image from camera.')
parser.add_argument('--camidx',
					type=int,
					default=0,
					help='camera index for cv2.VideoCapture')
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
Rx180 = np.array([[ 1, 0,  0],
				  [ 0 ,-1, 0],
				  [ 0 , 0, -1]])

#############################
# load opencv camera config #
#############################
video_stream = cv2.VideoCapture(args.camidx)
width, height = args.width, args.height

####################################
# getting opencv camera intrinsics #
####################################
intrin_name = args.intrin
if intrin_name[-5:] != '.txt':
	intrin_name = intrin_name + '.txt'
intrin_pth = os.path.join(os.getcwd(), 'configs', 'intrin', cfg_name)
K_full = np.loadtxt
K_full = np.loadtxt('configs/intrin/intrin_ps3.txt')
bbox = np.array([int(width/2-height/2), 0, width - int(width/2-height/2), height])

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
	_, bgr = video_stream.read()
	
	# process the frame
	bgr_crop, K_crop = utils.crop_img_by_bbox(bgr, bbox, K = K_full, crop_size = 512)
	markerCorners, markerIds, _ = cv2.aruco.detectMarkers(bgr_crop, charucodict)
	cv2.aruco.drawDetectedMarkers(bgr_crop, markerCorners, markerIds)
	
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
video_stream.release()
cv2.destroyAllWindows()
