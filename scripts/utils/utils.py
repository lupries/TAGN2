import yaml
import glob
import os
import numpy as np
import torch
import cv2
from ..datasets.dataloader_AGNN import create_dataloader
from ..datasets.dataloader import Resize
from torchvision import transforms

def generate_masks(model, root_dir, target_dir, imageFolder, img_size, batch_size, step):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloader = create_dataloader(root_dir, imageFolder, 'DAVIS/Annotations/480p/', img_size, subset='val',
                                   batch_size=batch_size, step=step)
    model.eval()
    image_names = {}
    info_file = open(root_dir + 'db_info.yaml', 'r')
    seq_list = yaml.load(info_file)['sequences']
    for seq in seq_list:
        if seq['year'] == 2016 and seq['set'] == 'val':
            seq_name = seq['name']
            image_names[seq_name] = sorted(os.listdir(os.path.join(root_dir, imageFolder, seq_name)))

    for keys in image_names:
        counter = 0
        os.mkdir(os.path.join(target_dir, keys))
        for samples in dataloader:
            inputs = samples['image'].to(device)
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
            img = image_names[keys][counter]
            counter += 1
            mask_img = outputs[0][0].data.cpu().numpy()
            retransform = Resize((854, 480), (854, 480))
            mask_img = retransform({'image': mask_img, 'mask': mask_img})
            mask_img = np.asarray(mask_img['image'] > 0.5, dtype=np.uint8)
            img_png = img[:-4] + '.png'
            cv2.imwrite(os.path.join(target_dir, keys, img_png), mask_img)
            if counter == len(image_names[keys]):
                break