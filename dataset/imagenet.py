import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import cv2
from datasets import load_dataset

class CustomDataset(Dataset):
    def __init__(self, feature_dir, label_dir, condition_dir=None, get_condition_img=False):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.flip = 'flip' in self.feature_dir
        self.get_condition_img = get_condition_img

        aug_feature_dir = feature_dir.replace('ten_crop/', 'ten_crop_105/')
        aug_label_dir = label_dir.replace('ten_crop/', 'ten_crop_105/')
        if os.path.exists(aug_feature_dir) and os.path.exists(aug_label_dir):
            self.aug_feature_dir = aug_feature_dir
            self.aug_label_dir = aug_label_dir
        else:
            self.aug_feature_dir = None
            self.aug_label_dir = None

        if condition_dir is not None:
            self.condition_dir = condition_dir
            self.aug_condition_dir = condition_dir.replace('ten_crop/', 'ten_crop_105/')
            if os.path.exists(self.aug_condition_dir):
                self.aug_condition_dir = self.aug_condition_dir
            else:
                self.aug_condition_dir = None
        else:
            self.condition_dir = None

        # file_num = min(129398,len(os.listdir(feature_dir)))
        file_num = len(os.listdir(feature_dir))
        # file_num = 1000
        self.feature_files = [f"{i}.npy" for i in range(file_num)]
        self.label_files = [f"{i}.npy" for i in range(file_num)]
        self.condition_files = [f"{i}.npy" for i in range(file_num)]
        # self.feature_files = sorted(os.listdir(feature_dir))
        # self.label_files = sorted(os.listdir(label_dir))
        # TODO: make it configurable
        # self.feature_files = [f"{i}.npy" for i in range(1281167)]
        # self.label_files = [f"{i}.npy" for i in range(1281167)]

    def __len__(self):
        assert len(self.feature_files) == len(self.label_files), \
            "Number of feature files and label files should be same"
        return len(self.feature_files)

    def __getitem__(self, idx):
        if self.aug_feature_dir is not None and torch.rand(1) < 0.5:
            feature_dir = self.aug_feature_dir
            label_dir = self.aug_label_dir
        else:
            feature_dir = self.feature_dir
            label_dir = self.label_dir
            if self.condition_dir is not None:
                condition_dir = self.condition_dir
                   
        feature_file = self.feature_files[idx]
        label_file = self.label_files[idx]
        if self.condition_dir is not None:
            condition_file = self.condition_files[idx]
            # condition_code = np.load(os.path.join(condition_dir, condition_file))
            condition_imgs = np.load(os.path.join(os.path.dirname(condition_dir), os.path.basename(condition_dir).replace('codes', 'imagesnpy'), condition_file))/255
            condition_imgs = 2*(condition_imgs-0.5)
            if self.get_condition_img:
                # print(os.path.join(os.path.dirname(condition_dir), os.path.basename(condition_dir).replace('codes', 'images'), condition_file.replace('npy', 'png')))
                condition_img = cv2.imread(os.path.join(os.path.dirname(condition_dir), os.path.basename(condition_dir).replace('codes', 'images'), condition_file.replace('npy', 'png')))/255
                condition_img = 2*(condition_img-0.5)
            #condition = condition[None,...].repeat(3, axis=2)

        features = np.load(os.path.join(feature_dir, feature_file))
        if self.flip:
            aug_idx = torch.randint(low=0, high=features.shape[1], size=(1,)).item()
            if self.get_condition_img:
                aug_idx = 0
            features = features[:, aug_idx]
            if self.condition_dir is not None:
                # condition_code = condition_code[:, aug_idx]
                condition_imgs = condition_imgs[aug_idx]
                
        labels = np.load(os.path.join(label_dir, label_file))
        # if self.condition_dir is not None:
        #     if self.get_condition_img:
        #         return torch.from_numpy(condition_img.transpose(2,0,1)).to(torch.float32), torch.from_numpy(condition)  # (1, 256), (1,1)
        #     else:
        #         return torch.from_numpy(features), torch.from_numpy(labels), torch.from_numpy(condition)  # (1, 256), (1,1)
        # else:
        #     return torch.from_numpy(features), torch.from_numpy(labels)
        outputs = {}
        outputs['img_code'] = torch.from_numpy(features)
        outputs['labels'] = torch.from_numpy(labels)
        if self.condition_dir is not None:
            # outputs['condition_code'] = torch.from_numpy(condition_code)
            outputs['condition_imgs'] = torch.from_numpy(condition_imgs)
        if self.get_condition_img:
            outputs['condition_img'] = torch.from_numpy(condition_img.transpose(2,0,1))
        return outputs


def build_imagenet(args, transform):
    return ImageFolder(args.data_path, transform=transform)

def build_imagenet_code(args):
    feature_dir = f"{args.code_path}/imagenet{args.image_size}_codes"
    label_dir = f"{args.code_path}/imagenet{args.image_size}_labels"
    if args.condition_type == 'canny':
        condition_dir = f"{args.code_path}/imagenet{args.image_size}_canny_codes"
    elif args.condition_type == 'hed':
        condition_dir = f"{args.code_path}/imagenet{args.image_size}_hed_codes"
    elif args.condition_type == 'depth':
        condition_dir = f"{args.code_path}/imagenet{args.image_size}_depth_codes"
    elif args.condition_type == 'none':
        condition_dir = None
    assert os.path.exists(feature_dir) and os.path.exists(label_dir), \
        f"please first run: bash scripts/autoregressive/extract_codes_c2i.sh ..."
    return CustomDataset(feature_dir, label_dir, condition_dir, args.get_condition_img)