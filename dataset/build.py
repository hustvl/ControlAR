from dataset.imagenet import build_imagenet, build_imagenet_code
from dataset.coco import build_coco
from dataset.openimage import build_openimage
from dataset.pexels import build_pexels
from dataset.t2i import build_t2i, build_t2i_code, build_t2i_image
from dataset.Edit_PIPE import PIPE_Dataset
from dataset.Edit_SeedxUnsplash import SeedxUnsplash_Dataset
from dataset.Edit_Pico import Pico_Dataset
# from dataset.t2i_control import build_t2i_control
import torch
import random

class Combine_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_args,
                 PIPE_Dataset=None,
                 SeedxUnsplash_Dataset=None,
                 Pico_Dataset=None,
                 mode='train',
                 ):
        self.dataset_args = dataset_args
        self.mode = mode
        self.total_len = 0

        self.PIPE_Dataset = PIPE_Dataset
        if PIPE_Dataset:
            self.total_len += len(PIPE_Dataset)
            self.pipe_len = len(PIPE_Dataset)
        else:
            self.pipe_len = 0
            
        self.SeedxUnsplash_Dataset = SeedxUnsplash_Dataset
        if SeedxUnsplash_Dataset:
            self.total_len += len(SeedxUnsplash_Dataset)
            self.seed_len = len(SeedxUnsplash_Dataset)
        else:
            self.seed_len = 0
        
        self.Pico_Dataset = Pico_Dataset
        if Pico_Dataset:
            self.total_len += len(Pico_Dataset)
            self.pico_len = len(Pico_Dataset)

        print("total number of images: ", self.total_len)

    def __getitem__(self, index):
        # ratio = random.random()

        # if self.PIPE_Dataset and ratio < self.pipe_len:
        #     index = random.randint(0, len(self.PIPE_Dataset) - 1)
        #     if self.mode == 'val':
        #         index = i
        #     PIPE_Dataset_data = self.PIPE_Dataset[index]
        #     return PIPE_Dataset_data
        # elif self.SeedxUnsplash_Dataset and ratio < self.dataset_args.seedxunsplash_prob:
        #     index = random.randint(0, len(self.SeedxUnsplash_Dataset) - 1)
        #     if self.mode == 'val':
        #         index = i
        #     SeedxUnsplash_Dataset_data = self.SeedxUnsplash_Dataset[index]
        #     return SeedxUnsplash_Dataset_data
        if index < self.pipe_len:
            PIPE_Dataset_data = self.PIPE_Dataset[index]
            return PIPE_Dataset_data
        elif index < self.pipe_len+self.seed_len:
            SeedxUnsplash_Dataset_data = self.SeedxUnsplash_Dataset[index-self.pipe_len]
            return SeedxUnsplash_Dataset_data
        else:
            Pico_Dataset_data = self.Pico_Dataset[index-self.pipe_len-self.seed_len]
            return Pico_Dataset_data
        
    def __len__(self):
        return self.total_len



def build_edit_dataset(dataset_args, llm_tokenizer, mode='train'):

    if 'pipe' in dataset_args.dataset_list:
        PIPE_Dataset_path= dataset_args.pipe_path
        PIPE_train_dataset = PIPE_Dataset(
            args=dataset_args,
            dataset_path=PIPE_Dataset_path,
            llm_tokenizer=llm_tokenizer,
            mode='train')
    else:
        PIPE_train_dataset = None

    if 'seedxunsplash' in dataset_args.dataset_list:
        SeedxUnsplash_Dataset_path= dataset_args.seedxunsplash_path
        SeedxUnsplash_train_dataset = SeedxUnsplash_Dataset(
            args=dataset_args,
            dataset_path=SeedxUnsplash_Dataset_path,
            llm_tokenizer=llm_tokenizer,
            mode='train')
    else:
        SeedxUnsplash_train_dataset = None
        
    if 'pico' in dataset_args.dataset_list:
        Pico_train_dataset = Pico_Dataset(
            args=dataset_args,
            llm_tokenizer=llm_tokenizer,
            mode='train')
    else:
        Pico_train_dataset = None

    train_dataset = Combine_Dataset(
                        dataset_args = dataset_args,
                        PIPE_Dataset = PIPE_train_dataset,
                        SeedxUnsplash_Dataset = SeedxUnsplash_train_dataset,
                        Pico_Dataset = Pico_train_dataset,
                        mode=mode,
                        )

    val_dataset = None
    return train_dataset



def build_dataset(args, **kwargs):
    # images
    if args.dataset == 'imagenet':
        return build_imagenet(args, **kwargs)
    if args.dataset == 'imagenet_code':
        return build_imagenet_code(args, **kwargs)
    if args.dataset == 'coco':
        return build_coco(args, **kwargs)
    if args.dataset == 'openimage':
        return build_openimage(args, **kwargs)
    if args.dataset == 'pexels':
        return build_pexels(args, **kwargs)
    if args.dataset == 't2i_image':
        return build_t2i_image(args, **kwargs)
    if args.dataset == 't2i':
        return build_t2i(args, **kwargs)
    if args.dataset == 't2i_code':
        return build_t2i_code(args, **kwargs)
    # if args.dataset == 't2i_control':
    #     return build_t2i_control(args, **kwargs)
    
    raise ValueError(f'dataset {args.dataset} is not supported')