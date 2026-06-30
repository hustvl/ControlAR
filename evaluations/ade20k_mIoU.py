import os
import numpy as np
from mmseg.apis import init_model, inference_model#, show_result_pyplot#, inference_segmentor
import torch
from PIL import Image
# from sklearn.metrics import confusion_matrix
from torchmetrics import JaccardIndex

def main():
    config_file = 'mmsegmentation/configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py'
    checkpoint_file = 'evaluations/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235933-7120c214.pth'
    
    # build the model from a config file and a checkpoint file
    model = init_model(config_file, checkpoint_file, device='cuda:0')
    
    # Image and segmentation labels directories
    # img_dir = 'sample/ade20k/visualization'
    img_dir = '/data/vjuicefs_ai_camera_album_ql/11171709/OmniGen/sample/ade20k'
    # img_dir = '/data/vjuicefs_ai_camera_album_ql/public_data/11171709/data/Captioned_ADE20K/val/image'
    # ann_dir = 'sample/ade20k/annotations'
    ann_dir = '/data/vjuicefs_ai_camera_album_ql/public_data/11171709/data/Captioned_ADE20K/val/label'
    
    # List all image files
    img_fns = [f for f in sorted(os.listdir(img_dir)) if f.endswith(".png")]
    # ann_fns = [f for f in sorted(os.listdir(ann_dir)) if f.endswith(".png")]
    
    total_mIoU = 0
    from tqdm import tqdm
    i = 0
    jaccard_index = JaccardIndex(task="multiclass", num_classes=150)
    num_classes = 150
    conf_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    for img_fn in tqdm(img_fns):
        i += 1
        # if i >= 100:
        #     break
        # try:
        img_path = os.path.join(img_dir, img_fn)
        # ann_path = os.path.join(ann_dir, img_fn)
        ann_path = os.path.join(ann_dir, f"{int(os.path.splitext(img_fn)[0])}.png")
        result = inference_model(model, img_path)
        # except Exception as e:
        #     continue
        # Read ground truth segmentation map
        gt_semantic_seg = np.array(Image.open(ann_path))

        ignore_label = 0
        gt = gt_semantic_seg.copy()
        pred = result.pred_sem_seg.data[0].cpu().numpy().copy()+1
        gt[gt == ignore_label] = num_classes
        try:
            conf_matrix += np.bincount(
                (num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=conf_matrix.size,
            ).reshape(conf_matrix.shape)
        except Exception as e:
            # import pdb;pdb.set_trace()
            # print(conf_matrix)
            pass
        
    
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