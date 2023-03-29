import cv2
import numpy as np
import os
import argparse
import json
from PIL import Image

#####################
# parse config json #
#####################
parser = argparse.ArgumentParser(description='Aruco Board Generator.')
parser.add_argument('--config',
					type=str,
					default='sl200_ml120',
					help='config json, located in configs/aruco/')
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

########################
# generate aruco board #
########################
charucodict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
img = cv2.aruco.drawCharucoDiamond(charucodict,np.array(ids),sl,ml)
img = Image.fromarray(img)
img.show()

############################
# save as png for printing #
############################
cfg_name = cfg_name[:-5] +'.png'
save_pth = os.path.join(os.getcwd(), 'out', 'aruco', cfg_name)
print(save_pth)
img.save(save_pth)
