import yaml
import glob
import os
import numpy as np
import torch
import cv2
from TAGN2.datasets.dataloader_AGNN import create_dataloader
from TAGN2.datasets.dataloader import Resize, ToTensor, Normalize
from torchvision import transforms
import matplotlib.pyplot as plt


def generate_masks(model, root_dir, target_dir, imageFolder, img_size, batch_size, step):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    transform = transforms.Compose([Resize(img_size, img_size), ToTensor(), Normalize()])
    info_file = open(root_dir + 'db_info.yaml', 'r')
    seq_list = yaml.load(info_file)['sequences']
    for seq in seq_list:
        if seq['year'] == 2016 and seq['set'] == 'val':
            seq_name = seq['name']
            os.mkdir(os.path.join(target_dir, seq_name))
            image_names = sorted(glob.glob(os.path.join(root_dir, imageFolder, seq_name, '*')))
            new_image_names = []
            for elem in range(len(image_names)):
                if elem + step * (batch_size - 1) < len(image_names):
                    for frame in range(batch_size):
                        new_image_names.append(image_names[elem + frame * step])
                else:
                    for frame in range(batch_size):
                        new_image_names.append(image_names[elem + frame * step - len(image_names)])
            image_names = new_image_names
            for i in range(int(len(image_names)/batch_size)):
                batch = torch.zeros(1, batch_size, 3, img_size[1], img_size[0])
                img = image_names[i*batch_size][-9:]
                for j in range(batch_size):
                    if i + j > int(len(image_names) / batch_size):
                       break
                    image = cv2.imread(image_names[i*batch_size + j], 1).transpose(2, 0, 1)
                    image_dict = {'image': image, 'mask': image}
                    image_dict = transform(image_dict)
                    batch[0, j] = image_dict['image']
                with torch.set_grad_enabled(False):
                    inputs = batch.to(device)
                    outputs = model(inputs)
                    mask_img = outputs[0][0].data.cpu().numpy()
                    mask_img = 1.0 / (1.0 + np.exp(-mask_img))
                    retransform = Resize((854, 480), (854, 480))
                    mask_img = retransform({'image': mask_img, 'mask': mask_img})
                    mask_img = np.asarray(mask_img['image'] > 0.5, dtype=np.uint8)
                    img_png = img[:-4] + '.png'
                    print(img_png)
                    cv2.imwrite(os.path.join(target_dir, seq_name, img_png), mask_img)
                    #plt.subplot(1, 1, 1)
                    #plt.imshow(mask_img)
                #plt.show(block=True)

