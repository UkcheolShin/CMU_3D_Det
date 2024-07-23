# Written by Ukcheol Shin (ushin@andrew.cmu.edu)
from argparse import ArgumentParser
from mmdet3d.apis import (inference_multi_modality_detector, init_model,
                          show_result_meshlab)
import warnings
warnings.filterwarnings("ignore")

def main():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('image', help='image file')
    parser.add_argument('calib', help='calibration file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo/results', help='dir to save results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()

    # raw sensor forwarding test
    # import mmcv
    # import numpy as np
    # args.image = mmcv.imread(args.image, 'unchanged')
    # args.pcd = np.fromfile(args.pcd, dtype=np.float32)

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    print('Model initialization done...')

    # test a single image
    result, data = inference_multi_modality_detector(model, args.pcd,
                                                     args.image, args.calib)
    """
    inference_multi_modality_detector
    Args:
        model (nn.Module): The loaded detector.
        pcd (str or pc): Point cloud files.
        image (str or numpy): Image files.
        calib_file (str): Calibration files.

    Returns:
        tuple: Predicted results and data from pipeline
        containing following results: ['boxes_3d', 'scores_3d', 'labels_3d']
        # 'boxes_3d'  : Each row is (x, y, z, x_size, y_size, z_size, yaw) in Lidar coodinates 
        #             : Nx7, where N stands for detected objects
        #             : (see details: mmdet3d/core/bbox/lidar_box3d.py)
        # 'scores_3d' : confidence score (Nx1), 0 is low, 1 is high confidence 
        # 'label_id'  : label of detected objects (Nx1), 0: pedestrian, 1: cylist, 2:car 
    """
    print('Inference done...')

    # show the results
    show_result_meshlab(
        data,
        result,
        args.out_dir,
        args.score_thr,
        # img=args.image, # uncomment when test raw sensor data
        snapshot=args.snapshot,
        task='multi_modality-det')

    print('Prediction result is saved...')

if __name__ == '__main__':
    main()
