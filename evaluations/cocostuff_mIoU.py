import os
import numpy as np
from mmseg.apis import init_model, inference_model, show_result_pyplot#, inference_segmentor
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
from torchmetrics import JaccardIndex

def main():
    config_file = 'mmsegmentation/configs/deeplabv3/deeplabv3_r101-d8_4xb4-320k_coco-stuff164k-512x512.py'
    checkpoint_file = 'evaluations/deeplabv3_r101-d8_512x512_4x4_320k_coco-stuff164k_20210709_155402-3cbca14d.pth'
    
    # build the model from a config file and a checkpoint file
    model = init_model(config_file, checkpoint_file, device='cuda:1')
    
    # Image and segmentation labels directories
    img_dir = 'sample/cocostuff/visualization'
    ann_dir = 'sample/cocostuff/annotations'
    
    # List all image files
    img_fns = [f for f in sorted(os.listdir(img_dir)) if f.endswith(".png")]

    
    total_mIoU = 0
    from tqdm import tqdm
    i = 0
    num_classes = 171
    jaccard_index = JaccardIndex(task="multiclass", num_classes=num_classes)
    
    conf_matrix = np.zeros((num_classes+1, num_classes+1), dtype=np.int64)
    for img_fn in tqdm(img_fns):
        ann_fn = img_fn
        i += 1
        # if i == 4891:
        #     continue
        try:
            img_path = os.path.join(img_dir, img_fn)
            ann_path = os.path.join(ann_dir, img_fn)
            result = inference_model(model, img_path)
        except Exception as e:
            continue
        # Read ground truth segmentation map
        gt_semantic_seg = np.array(Image.open(ann_path))

        ignore_label = 255
        gt = gt_semantic_seg.copy()
        # import pdb;pdb.set_trace()
        # print(np.unique(gt), np.unique(result.pred_sem_seg.data[0].cpu().numpy()))
        pred = result.pred_sem_seg.data[0].cpu().numpy().copy()#+1
        gt[gt == ignore_label] = num_classes
        conf_matrix += np.bincount(
            (num_classes+1) * pred.reshape(-1) + gt.reshape(-1),
            minlength=conf_matrix.size,
        ).reshape(conf_matrix.shape)
        
    
    # calculate miou
    acc = np.full(num_classes, np.nan, dtype=np.float64)
    iou = np.full(num_classes, np.nan, dtype=np.float64)
    tp = conf_matrix.diagonal()[:-1].astype(np.float64)
    pos_gt = np.sum(conf_matrix[:-1, :-1], axis=0).astype(np.float64)
    pos_pred = np.sum(conf_matrix[:-1, :-1], axis=1).astype(np.float64)
    acc_valid = pos_gt > 0
    acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
    iou_valid = (pos_gt + pos_pred) > 0
    union = pos_gt + pos_pred - tp
    iou[acc_valid] = tp[acc_valid] / union[acc_valid]
    miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
    print(f"mIoU: {miou}")

if __name__ == '__main__':
    main()