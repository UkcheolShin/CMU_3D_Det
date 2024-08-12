import copy
import cv2
import numpy as np
import torch
from os import path as osp
import colorsys
import mmcv

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / float(N), 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    return colors

max_color = 30
colors = random_colors(max_color)       # Generate random colors

def plot_rect3d_on_img(img,
                       rect_corners,
                       color=(0, 255, 0),
                       thickness=3):
    """Plot the boundary lines of 3D rectangular on 2D images.

    Args:
        img (numpy.array): The numpy array of image.
        rect_corners (numpy.array): Coordinates of the corners of 3D
            rectangulars. Should be in the shape of [8, 2].
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
    corners = rect_corners.astype(int)
    for start, end in line_indices:
        cv2.line(img, (corners[start, 0], corners[start, 1]),
                 (corners[end, 0], corners[end, 1]), color, thickness,
                 cv2.LINE_AA)

    return img.astype(np.uint8)

def proj_lidar_bbox3d_on_img(bboxes3d,
                             lidar2img_rt):
    """Project the 3D bbox on 2D image plane.

    Args:
        bboxes3d (:obj:`LiDARInstance3DBoxes`):
            3d bbox in lidar coordinate system to visualize.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
    """
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3),
         np.ones((num_bbox * 8, 1))], axis=-1)
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    pts_2d = pts_4d @ lidar2img_rt.T

    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    return imgfov_pts_2d

def save_image_with_boxes(img_filename,
						  obj_bboxes2d,
						  obj_ids, 
						  obj_scores, 
						  result_path, 
						  score_thr=0.0):
    """Draw the 3D bbox on 2D image plane.
        obj_bboxes2d: (N,8,2) array of vertices for the 3d box in following order:
            6 -------- 5 
           /|         /|
          2 -------- 1 .
          | |        | |
          . 7 -------- 4
          |/         |/
          3 -------- 0   
    """
    img = mmcv.imread(img_filename)
    file_name = osp.split(img_filename)[-1].split('.')[0]
    
    for obj_id, obj_bbox_2d, obj_score in zip(obj_ids, obj_bboxes2d, obj_scores):
        if obj_score < score_thr:
            continue
        color_tmp = tuple([int(tmp * 255) for tmp in colors[obj_id % max_color]])
        img = plot_rect3d_on_img(img, obj_bbox_2d, color=color_tmp)
        text = 'ID: %d' % obj_id
        img = cv2.putText(img, text, (int(obj_bbox_2d[6, 0]), int(obj_bbox_2d[6, 1]) - 8), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color=color_tmp) 
    mmcv.imwrite(img, osp.join(result_path, f'{file_name}_pred.png'))
