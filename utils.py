import cv2
import numpy as np

################################################
# most of the util functions here are imported #
# from the original OnePose repository         #
################################################

def reproj(K, pose, pts_3d):
    """ 
    Reproj 3d points to 2d points 
    @param K: [3, 3] or [3, 4]
    @param pose: [3, 4] or [4, 4]
    @param pts_3d: [n, 3]
    """
    assert K.shape == (3, 3) or K.shape == (3, 4)
    assert pose.shape == (3, 4) or pose.shape == (4, 4)

    if K.shape == (3, 3):
        K_homo = np.concatenate([K, np.zeros((3, 1))], axis=1)
    else:
        K_homo = K
    
    if pose.shape == (3, 4):
        pose_homo = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
    else:
        pose_homo = pose
    
    pts_3d = pts_3d.reshape(-1, 3)
    pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1))], axis=1)
    pts_3d_homo = pts_3d_homo.T

    reproj_points = K_homo @ pose_homo @ pts_3d_homo
    reproj_points = reproj_points[:] / reproj_points[2:]
    reproj_points = reproj_points[:2, :].T
    return reproj_points # [n, 2]
    
def draw_3d_box(image, corners_2d, linewidth=3, color='g'):
    """ Draw 3d box corners 
    @param corners_2d: [8, 2]
    """
    lines = np.array([
        [0, 1, 5, 4, 2, 3, 7, 6, 2, 2, 3, 7],
        [1, 5, 4, 0, 3, 0, 6, 5, 1, 6, 7, 4]
    ]).T

    colors = {
        'g': (0, 255, 0),
        'r': (0, 0, 255),
        'b': (255, 0, 0)
    }
    if color not in colors.keys():
        color = (42, 97, 247)
    else:
        color = colors[color]
    
    for id, line in enumerate(lines):
        pt1 = corners_2d[line[0]].astype(int)
        pt2 = corners_2d[line[1]].astype(int)
        cv2.line(image, tuple(pt1), tuple(pt2), color, linewidth)

def rect_to_square(self, x0, y0, x1, y1):
	w, h = (x1 - x0), (y1 - y0)
	dw, dh = w/2, h/2
	if h > w:
		x0 = int(x0 + dw - dh)
		x1 = int(x1 - dw + dh)
	else:
		y0 = int(y0 + dh - dw)
		y1 = int(y1 - dh + dw)
	return x0, y0, x1, y1
	
def get_dir(src_point, rot_rad):
	sn, cs = np.sin(rot_rad), np.cos(rot_rad)

	src_result = [0, 0]
	src_result[0] = src_point[0] * cs - src_point[1] * sn
	src_result[1] = src_point[0] * sn + src_point[1] * cs

	return src_result
	
def get_3rd_point(a, b):
	direct = a - b
	return b + np.array([-direct[1], direct[0]], dtype=np.float32)
	
def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
	if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
		scale = np.array([scale, scale], dtype=np.float32)

	scale_tmp = scale
	src_w = scale_tmp[0]
	dst_w = output_size[0]
	dst_h = output_size[1]

	rot_rad = np.pi * rot / 180
	src_dir = get_dir([0, src_w * -0.5], rot_rad)
	dst_dir = np.array([0, dst_w * -0.5], np.float32)

	src = np.zeros((3, 2), dtype=np.float32)
	dst = np.zeros((3, 2), dtype=np.float32)
	src[0, :] = center + scale_tmp * shift
	src[1, :] = center + src_dir + scale_tmp * shift
	dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
	dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

	src[2:, :] = get_3rd_point(src[0, :], src[1, :])
	dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

	if inv:
		trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
	else:
		trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

	return trans
	
def get_K_crop_resize(box, K_orig, resize_shape):
	"""Update K (crop an image according to the box, and resize the cropped image to resize_shape) 
	@param box: [x0, y0, x1, y1]
	@param K_orig: [3, 3] or [3, 4]
	@resize_shape: [h, w]
	"""
	center = np.array([(box[0] + box[2]) / 2., (box[1] + box[3]) / 2.])
	scale = np.array([box[2] - box[0], box[3] - box[1]]) # w, h

	resize_h, resize_w = resize_shape
	trans_crop = get_affine_transform(center, scale, 0, [resize_w, resize_h])
	trans_crop_homo = np.concatenate([trans_crop, np.array([[0, 0, 1]])], axis=0)

	if K_orig.shape == (3, 3):
		K_orig_homo = np.concatenate([K_orig, np.zeros((3, 1))], axis=-1)
	else:
		K_orig_homo = K_orig.copy()
	assert K_orig_homo.shape == (3, 4)

	K_crop_homo = trans_crop_homo @ K_orig_homo # [3, 4]
	K_crop = K_crop_homo[:3, :3]

	return K_crop, K_crop_homo
    
def get_image_crop_resize(image, box, resize_shape):
	"""Crop image according to the box, and resize the cropped image to resize_shape
	@param image: the image waiting to be cropped
	@param box: [x0, y0, x1, y1]
	@param resize_shape: [h, w]
	"""
	center = np.array([(box[0] + box[2]) / 2., (box[1] + box[3]) / 2.])
	scale = np.array([box[2] - box[0], box[3] - box[1]])

	resize_h, resize_w = resize_shape
	trans_crop = get_affine_transform(center, scale, 0, [resize_w, resize_h])
	image_crop = cv2.warpAffine(image, trans_crop, (resize_w, resize_h), flags=cv2.INTER_LINEAR)

	trans_crop_homo = np.concatenate([trans_crop, np.array([[0, 0, 1]])], axis=0)
	return image_crop, trans_crop_homo
	
def crop_img_by_bbox(origin_img, bbox, K=None, crop_size=512):
	"""
	Crop image by detect bbox
	Input:
		query_img_path: str,
		bbox: np.ndarray[x0, y0, x1, y1],
		K[optional]: 3*3
	Output:
		image_crop: np.ndarray[crop_size * crop_size],
		K_crop[optional]: 3*3
	"""
	x0, y0 = bbox[0], bbox[1]
	x1, y1 = bbox[2], bbox[3]

	resize_shape = np.array([y1 - y0, x1 - x0])
	if K is not None:
		K_crop, K_crop_homo = get_K_crop_resize(bbox, K, resize_shape)
	image_crop, trans1 = get_image_crop_resize(origin_img, bbox, resize_shape)


	bbox_new = np.array([0, 0, x1 - x0, y1 - y0])
	resize_shape = np.array([crop_size, crop_size])
	if K is not None:
		K_crop, K_crop_homo = get_K_crop_resize(bbox_new, K_crop, resize_shape)
	image_crop, trans2 = get_image_crop_resize(image_crop, bbox_new, resize_shape)
	
	return image_crop, K_crop if K is not None else None
