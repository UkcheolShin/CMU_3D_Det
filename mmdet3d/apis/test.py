# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp

import mmcv
import torch
from mmcv.image import tensor2imgs

from mmdet3d.models import (Base3DDetector, Base3DSegmentor,
                            SingleStageMono3DDetector)

from mmdet3d.core import bbox3d2result

def single_gpu_test(model, pred2bbox, 
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool, optional): Whether to save viualization results.
            Default: True.
        out_dir (str, optional): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model = model.to('cuda')
    pred2bbox = pred2bbox.to('cuda')
    model.eval()
    pred2bbox.eval()

    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # result = model(return_loss=False, rescale=True, **data)
            result = model(data['voxels'][0].data[0].squeeze(0),\
                           data['num_points'][0].data[0].squeeze(0),\
                           data['coors'][0].data[0].squeeze(0),\
                           data['img'][0].data[0].squeeze(0),\
                           torch.from_numpy(data['img_metas'][0].data[0][0]['lidar2img']))
            # result = model(data['voxels'][0].data[0].squeeze(0), data['num_points'][0].data[0].squeeze(0),                           data['coors'][0].data[0].squeeze(0),                           data['img'][0].data[0].squeeze(0),torch.from_numpy(data['img_metas'][0].data[0][0]['lidar2img']))
            # result2 = pred2bbox(data['voxels'][0].data[0].squeeze(0), data['num_points'][0].data[0].squeeze(0),                           data['coors'][0].data[0].squeeze(0),                           data['img'][0].data[0].squeeze(0),torch.from_numpy(data['img_metas'][0].data[0][0]['lidar2img']))

            img_metas = data['img_metas'][0].data
            bbox_list = pred2bbox.get_bboxes(*result, img_metas[0], rescale=True)
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in bbox_list
            ]

            bbox_list_out = [dict() for i in range(len(img_metas))]
            for result_dict, pts_bbox in zip(bbox_list_out, bbox_results):
                result_dict['pts_bbox'] = pts_bbox
            result =  bbox_list_out
        if show:
            # Visualize the results of MMDetection3D model
            # 'show_results' is MMdetection3D visualization API
            models_3d = (Base3DDetector, Base3DSegmentor,
                         SingleStageMono3DDetector)
            if isinstance(model.module, models_3d):
                model.module.show_results(
                    data,
                    result,
                    out_dir=out_dir,
                    show=show,
                    score_thr=show_score_thr)
            # Visualize the results of MMDetection model
            # 'show_result' is MMdetection visualization API
            else:
                batch_size = len(result)
                if batch_size == 1 and isinstance(data['img'][0],
                                                  torch.Tensor):
                    img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results
