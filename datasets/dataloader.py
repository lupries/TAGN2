import glob
import yaml
import os
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import cv2
import torch
from torchvision import transforms, utils
import numpy as np

class SegDataset(Dataset):
    """Segmentation Dataset"""

    def __init__(self, root_dir, imageFolder, maskFolder, transform=None, seed=None, fraction=None, subset=None, imagecolormode='rgb', maskcolormode='grayscale', batch_size=3, step=5):
        """
        Args:
            root_dir (string): Directory with all the images and should have the following structure.
            root
            --Images
            -----Img 1
            -----Img N
            --Mask
            -----Mask 1
            -----Mask N
            imageFolder (string) = 'Images' : Name of the folder which contains the Images.
            maskFolder (string)  = 'Masks : Name of the folder which contains the Masks.
            transform (callable, optional): Optional transform to be applied on a sample.
            seed: Specify a seed for the train and test split
            fraction: A float value from 0 to 1 which specifies the validation split fraction
            subset: 'Train' or 'Test' to select the appropriate set.
            imagecolormode: 'rgb' or 'grayscale'
            maskcolormode: 'rgb' or 'grayscale'
        """
        self.color_dict = {'rgb': 1, 'grayscale': 0}
        assert(imagecolormode in ['rgb', 'grayscale'])
        assert(maskcolormode in ['rgb', 'grayscale'])

        self.imagecolorflag = self.color_dict[imagecolormode]
        self.maskcolorflag = self.color_dict[maskcolormode]
        self.root_dir = root_dir
        self.transform = transform
        seq_file = open(root_dir+'db_info.yaml','r')
        sequences = yaml.load(seq_file)['sequences']
        self.sequences = [seq['name'] for seq in sequences if (seq['year']==2016 and subset == seq['set'])]
        print(self.sequences)
        self.image_names = []
        self.mask_names = []
        if not fraction:
            for seq in self.sequences:
                image_names = sorted(
                    glob.glob(os.path.join(self.root_dir, imageFolder, seq, '*')))
                mask_names = sorted(
                    glob.glob(os.path.join(self.root_dir, maskFolder, seq, '*')))
                if len(mask_names)<len(image_names):
                    print(mask_names)
                    mask = mask_names[0]
                    mask_names = [mask for i in range(len(image_names))]
                    print(mask_names)
                new_image_names = []
                new_mask_names = []
                for elem in range(len(image_names)):
                    if elem + step * (batch_size-1) < len(image_names):
                        for frame in range(batch_size):
                            new_image_names.append(image_names[elem + frame * step])
                            new_mask_names.append(mask_names[elem + frame * step])
                    else:
                        break
                self.image_names += new_image_names
                self.mask_names += new_mask_names
            new_image_names = []
            new_mask_names = []
            indices = np.arange(0, len(self.image_names), batch_size)
            np.random.shuffle(indices)
            for elem in indices:
                for i in range(0, batch_size):
                    if elem + i < len(self.image_names):
                        new_image_names.append(self.image_names[elem + i])
                        new_mask_names.append(self.mask_names[elem + i])
            self.image_names = new_image_names
            self.mask_names = new_mask_names
        else:
            assert(subset in ['Train', 'Test'])
            self.fraction = fraction
            self.image_list = []
            self.mask_list = []
            for seq in self.sequences:
                image_list = np.array(
                    sorted(glob.glob(os.path.join(self.root_dir, imageFolder, seq, '*'))))
                mask_list = np.array(
                    sorted(glob.glob(os.path.join(self.root_dir, maskFolder, seq, '*'))))
                self.image_list += image_list
                self.mask_list += mask_list
            if seed:
                np.random.seed(seed)
                indices = np.arange(len(self.image_list))
                np.random.shuffle(indices)
                self.image_list = self.image_list[indices]
                self.mask_list = self.mask_list[indices]
            if subset == 'Train':
                self.image_names = self.image_list[:int(
                    np.ceil(len(self.image_list)*(1-self.fraction)))]
                self.mask_names = self.mask_list[:int(
                    np.ceil(len(self.mask_list)*(1-self.fraction)))]
            else:
                self.image_names = self.image_list[int(
                    np.ceil(len(self.image_list)*(1-self.fraction))):]
                self.mask_names = self.mask_list[int(
                    np.ceil(len(self.mask_list)*(1-self.fraction))):]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        if self.imagecolorflag:
            image = cv2.imread(
                img_name, self.imagecolorflag).transpose(2, 0, 1)
        else:
            image = cv2.imread(img_name, self.imagecolorflag)
        msk_name = self.mask_names[idx]
        if self.maskcolorflag:
            mask = cv2.imread(msk_name, self.maskcolorflag).transpose(2, 0, 1)
        else:
            mask = cv2.imread(msk_name, self.maskcolorflag)
        mask = np.asarray(mask>0).astype(float)
        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Resize(object):
    """Resize image and/or masks."""

    def __init__(self, imageresize, maskresize):
        self.imageresize = imageresize
        self.maskresize = maskresize

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if len(image.shape) == 3:
            image = image.transpose(1, 2, 0)
        if len(mask.shape) == 3:
            mask = mask.transpose(1, 2, 0)
        mask = cv2.resize(mask, self.maskresize, cv2.INTER_AREA)
        image = cv2.resize(image, self.imageresize, cv2.INTER_AREA)
        if len(image.shape) == 3:
            image = image.transpose(2, 0, 1)
        if len(mask.shape) == 3:
            mask = mask.transpose(2, 0, 1)

        return {'image': image,
                'mask': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, maskresize=None, imageresize=None):
        image, mask = sample['image'], sample['mask']
        if len(mask.shape) == 2:
            mask = mask.reshape((1,)+mask.shape)
        if len(image.shape) == 2:
            image = image.reshape((1,)+image.shape)
        return {'image': torch.from_numpy(image),
                'mask': torch.from_numpy(mask)}


class Normalize(object):
    '''Normalize image'''

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        return {'image': image.type(torch.FloatTensor)/255,
                'mask': mask.type(torch.FloatTensor)}


def create_dataloader(data_dir, imageFolder, maskFolder, size = (256,256), fraction=None, subset='train', batch_size=4, step=5):

    data_transforms = transforms.Compose([Resize(size, size), ToTensor(), Normalize()])

    image_dataset = SegDataset(data_dir, transform=data_transforms, imageFolder=imageFolder, maskFolder=maskFolder, subset=subset, batch_size=batch_size, step=step)
    sampler = SequentialSampler(image_dataset)
    dataloader = DataLoader(image_dataset, sampler=sampler, batch_size=batch_size, num_workers=8)

    return dataloader

