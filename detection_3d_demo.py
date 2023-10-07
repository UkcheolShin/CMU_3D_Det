# Written by Ukcheol Shin (ushin@andrew.cmu.edu)
from argparse import ArgumentParser
from mmdet3d.apis import (inference_multi_modality_detector, init_model,
                          show_result_meshlab)

def main():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('image', help='image file')
    parser.add_argument('ann', help='ann file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
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
                                                     args.image, args.ann)
    print('Inference done...')

    # show the results
    show_result_meshlab(
        data,
        result,
        args.out_dir,
        args.score_thr,
        snapshot=args.snapshot,
        task='multi_modality-det')

    print('Prediction result is saved...')

if __name__ == '__main__':
    main()
