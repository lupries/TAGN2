import glob
import yaml
import os
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import cv2
import torch
from torchvision import transforms, utils
import numpy as np
from .dataloader import SegDataset

class SegDataset_AGNN(SegDataset):
  def __len__(self):
      return int(len(self.image_names)/self.batch_size)

  def __getitem__(self, idx):
      img_names = self.image_names[idx*self.batch_size:(idx+1)*self.batch_size]
      images = []
      for img_name in img_names:
        if self.imagecolorflag:
            image = cv2.imread(
                img_name, self.imagecolorflag).transpose(2, 0, 1)
        else:
            image = cv2.imread(img_name, self.imagecolorflag)
        images.append(image)
      
      msk_names = self.mask_names[idx*self.batch_size:(idx+1)*self.batch_size]
      masks = []
      for msk_name in msk_names:
        if self.maskcolorflag:
            mask = cv2.imread(msk_name, self.maskcolorflag).transpose(2, 0, 1)
        else:
            mask = cv2.imread(msk_name, self.maskcolorflag)
        mask = np.asarray(mask>0).astype(float)
        masks.append(mask)
      sample = {'image': np.asarray(images), 'mask': np.asarray(masks)}

      if self.transform:
          sample = self.transform(sample)

      return sample

class Resize(object):
    """Resize image and/or masks."""

    def __init__(self, imageresize, maskresize):
        self.imageresize = imageresize
        self.maskresize = maskresize

    def __call__(self, sample):
        images, masks = sample['image'], sample['mask']
        if len(images.shape) == 4:
            images = images.transpose(0, 2, 3, 1)
        if len(masks.shape) == 4:
            masks = masks.transpose(0, 2, 3, 1)
        new_masks = np.zeros((masks.shape[0], self.maskresize[0], self.maskresize[1]))
        new_images = np.zeros((images.shape[0], self.imageresize[0], self.imageresize[1], images.shape[3]))
        for i in range(images.shape[0]):
          new_masks[i] = cv2.resize(masks[i], self.maskresize, cv2.INTER_AREA)
          new_images[i] = cv2.resize(images[i], self.imageresize, cv2.INTER_AREA)
        if len(new_images.shape) == 4:
            new_images = new_images.transpose(0, 3, 1, 2)
        if len(new_masks.shape) == 4:
            new_masks = new_masks.transpose(0, 3, 1, 2)

        return {'image': new_images,
                'mask': new_masks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, maskresize=None, imageresize=None):
        images, masks = sample['image'], sample['mask']
        if len(masks.shape) == 3:
            masks = mask.reshape(masks.shape[0]+(1,)+masks.shape[1:])
        if len(images.shape) === 2:
            images = images.reshape(images.shape[0]+(1,)+images.shape[1:])
        return {'image': torch.from_numpy(images),
                'mask': torch.from_numpy(masks)}


class Normalize(object):
    '''Normalize image'''

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        return {'image': image.type(torch.FloatTensor)/255,
                'mask': mask.type(torch.FloatTensor)}


def create_dataloader(data_dir, imageFolder, maskFolder, size = (256,256), fraction=None, subset='train', batch_size=4, step=5):

    data_transforms = transforms.Compose([Resize(size, size), ToTensor(), Normalize()])

    image_dataset = SegDataset_AGNN(data_dir, transform=data_transforms, imageFolder=imageFolder, maskFolder=maskFolder, subset=subset, batch_size=batch_size, step=step)
    sampler = SequentialSampler(image_dataset)
    dataloader = DataLoader(image_dataset, sampler=sampler, batch_size=batch_size, num_workers=8)

    return dataloader

