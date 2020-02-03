import yaml
import glob
import os
import numpy as np
import torch
import cv2
from TAGN2.datasets.dataloader_AGNN import create_dataloader
from TAGN2.datasets.dataloader import Resize, ToTensor, Normalize
from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import copy
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


def oneshot_baseline_validation(model, criterion, optimizer, metrics, root_dir, target_dir, iterations, imageFolder,
                                maskFolder,
                                img_size, batch_size):
    writer = SummaryWriter('drive/My Drive/oneshot/runs/SGD_1e-4_10iter_last_layer')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    n_iter = 0
    # cache initial model weights
    initial_state = copy.deepcopy(model.state_dict())

    transform = transforms.Compose([Resize(img_size, img_size), ToTensor(), Normalize()])
    info_file = open(root_dir + 'db_info.yaml', 'r')
    seq_list = yaml.load(info_file)['sequences']
    for seq in seq_list:
        if seq['year'] == 2016 and seq['set'] == 'val':
            seq_name = seq['name']
            print(seq_name)
            os.mkdir(os.path.join(target_dir, seq_name))

            # load Image names and create batch of first frame
            image_names = sorted(glob.glob(os.path.join(root_dir, imageFolder, seq_name, '*')))
            mask_names = sorted(glob.glob(os.path.join(root_dir, maskFolder, seq_name, '*')))
            img_batch = torch.zeros(batch_size, 3, img_size[1], img_size[0])
            mask_batch = torch.zeros(batch_size, 1, img_size[1], img_size[0])
            for j in range(batch_size):
                image = cv2.imread(image_names[0], 1).transpose(2, 0, 1)
                mask = cv2.imread(mask_names[0], 0)
                mask = np.asarray(mask > 0).astype(float)
                image_dict = {'image': image, 'mask': mask}
                image_dict = transform(image_dict)
                img_batch[j] = image_dict['image']
                mask_batch[j] = image_dict['mask']
            inputs = img_batch.to(device)
            mask = mask_batch.to(device)

            # do online learning on first frame
            model.train()
            model.load_state_dict(initial_state)
            new_model_state = online_baseline_loop(inputs, mask, iterations, model, criterion, optimizer)
            model.load_state_dict(new_model_state)
            new_state = copy.deepcopy(model.state_dict())

            # generate outputs for the whole sequence
            model.eval()
            for i in range(int(len(image_names) / batch_size) + 1):
                n_iter += 1
                img = []
                img_batch = torch.zeros(batch_size, 3, img_size[1], img_size[0])
                mask_batch = torch.zeros(batch_size, 1, img_size[1], img_size[0])
                for j in range(batch_size):
                    if i * batch_size + j == int(len(image_names)):
                        break
                    img.append(image_names[i * batch_size + j][-9:])
                    image = cv2.imread(image_names[i * batch_size + j], 1).transpose(2, 0, 1)
                    mask = cv2.imread(mask_names[i * batch_size + j], 0)
                    mask = np.asarray(mask > 0).astype(float)
                    image_dict = {'image': image, 'mask': mask}
                    image_dict = transform(image_dict)
                    img_batch[j] = image_dict['image']
                    mask_batch[j] = image_dict['mask']
                inputs = img_batch.to(device)
                masks = mask_batch.to(device)
                model.load_state_dict(new_state)
                with torch.set_grad_enabled(False):
                    outputs = model(inputs)['out']
                    for j in range(int(outputs.shape[0])):
                        if i * batch_size + j == int(len(image_names)):
                            break
                        mask_img = outputs[j].data.cpu().numpy()
                        mask_img = 1.0 / (1.0 + np.exp(-mask_img))
                        retransform = Resize((854, 480), (854, 480))
                        mask_img = retransform({'image': mask_img, 'mask': mask_img})
                        mask_img = np.asarray(mask_img['image'] > 0.5, dtype=np.uint8)
                        img_png = img[j][:-4] + '.png'
                        cv2.imwrite(os.path.join(target_dir, seq_name, img_png), mask_img)

                # classify batch with refined model
                loss_refined, score_refined = classify_baseline_sequence(inputs, masks, model, criterion, metrics)
                # classify batch with unrefined model (comparison)
                model.load_state_dict(initial_state)
                loss_unrefined, score_unrefined = classify_baseline_sequence(inputs, masks, model, criterion, metrics)
                # log to tensorboard
                writer.add_scalars('Accuracy/f1_score', {'refined': score_refined,
                                                         'unrefined': score_unrefined,
                                                         'gain': score_refined - score_unrefined}, n_iter)
                writer.add_scalars('Loss', {'refined': loss_refined,
                                            'unrefined': loss_unrefined,
                                            'gain': loss_refined - loss_unrefined}, n_iter)
    writer.close()
    model.load_state_dict(initial_state)


def online_baseline_loop(inputs, masks, iterations, model, criterion, optimizer):
    writer = SummaryWriter('online_plots')
    for i in range(iterations):
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)['out']
            loss = criterion(outputs, masks)
            print('online loss:' + str(loss))
            loss.backward()
            optimizer.step()
            writer.add_scalars('Loss', {'loss': loss}, i)
    writer.close()
    return model.state_dict()


def classify_baseline_sequence(frames, masks, model, criterion, metrics):
    with torch.set_grad_enabled(False):
        outputs = model(frames)['out']
        loss = criterion(outputs, masks)
        y_pred = 1.0 / (1 + np.exp(-outputs.data.cpu().numpy().ravel()))
        y_true = masks.data.cpu().numpy().ravel()
        for name, metric in metrics.items():
            if name == 'f1_score':
                score = metric(y_true > 0, y_pred > 0.5)
    return loss, score


def standard_oneshot_validation(model, criterion, optimizer, metrics, root_dir, target_dir, iterations, imageFolder,
                                maskFolder, img_size, batch_size, step):
    writer = SummaryWriter('drive/My Drive/oneshot/runs/TAGNN_SGD_1e-5_5iter_ASPP')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    n_iter = 0
    # cache initial model weights
    initial_state = copy.deepcopy(model.state_dict())

    transform = transforms.Compose([Resize(img_size, img_size), ToTensor(), Normalize()])
    info_file = open(root_dir + 'db_info.yaml', 'r')
    seq_list = yaml.load(info_file)['sequences']
    for seq in seq_list:
        if seq['year'] == 2016 and seq['set'] == 'val':
            seq_name = seq['name']
            print(seq_name)
            os.mkdir(os.path.join(target_dir, seq_name))
            image_names = sorted(glob.glob(os.path.join(root_dir, imageFolder, seq_name, '*')))
            new_image_names = []
            mask_names = sorted(glob.glob(os.path.join(root_dir, maskFolder, seq_name, '*')))
            new_mask_names = []

            # create sequence dataset {{x, x+10, x+20},{x+1, x+1+10, x+1+20},...}
            for elem in range(len(image_names)):
                if elem + step * (batch_size - 1) < len(image_names):
                    for frame in range(batch_size):
                        new_image_names.append(image_names[elem + frame * step])
                        new_mask_names.append(mask_names[elem + frame*step])
                else:
                    for frame in range(batch_size):
                        new_image_names.append(image_names[elem + frame * step - len(image_names)])
                        new_mask_names.append(mask_names[elem + frame * step - len(image_names)])
            image_names = new_image_names
            mask_names = new_mask_names

            # create batch for online training, frames: {{0,10,20},{0,10,20},{0,10,20}}
            img_batch = torch.zeros(3, batch_size, 3, img_size[1], img_size[0])
            mask_batch = torch.zeros(3, batch_size, 1, img_size[1], img_size[0])
            for i in range(img_batch.shape[0]):
                for j in range(batch_size):
                    image = cv2.imread(image_names[j], 1).transpose(2, 0, 1)
                    mask = cv2.imread(mask_names[0], 0)
                    mask = np.asarray(mask > 0).astype(float)
                    image_dict = {'image': image, 'mask': mask}
                    image_dict = transform(image_dict)
                    img_batch[i, j] = image_dict['image']
                    mask_batch[i, j] = image_dict['mask']
            inputs = img_batch.to(device)
            mask = mask_batch.to(device)

            # do oneshot training
            model.train()
            model.load_state_dict(initial_state)
            new_model_state = standard_online_loop(inputs, mask, iterations, model, criterion, optimizer)
            model.load_state_dict(new_model_state)
            new_state = copy.deepcopy(model.state_dict())
            model.eval()

            # classify the rest of the sequence
            for i in range(int(len(image_names) / (batch_size*batch_size)) + 1):
                n_iter += 1
                img = []

                # create batch of data
                img_batch = torch.zeros(3, batch_size, 3, img_size[1], img_size[0])
                mask_batch = torch.zeros(3, batch_size, 1, img_size[1], img_size[0])
                for k in range(img_batch.shape[0]):
                    for j in range(batch_size):
                        if i * batch_size*batch_size + k * batch_size + j >= int(len(image_names)):
                            break
                        img.append(image_names[i * batch_size*batch_size + k*batch_size + j][-9:])
                        image = cv2.imread(image_names[i * batch_size*batch_size + k*batch_size + j], 1).transpose(2, 0, 1)
                        mask = cv2.imread(mask_names[i * batch_size*batch_size + k*batch_size + j], 0)
                        mask = np.asarray(mask > 0).astype(float)
                        image_dict = {'image': image, 'mask': mask}
                        image_dict = transform(image_dict)
                        img_batch[k, j] = image_dict['image']
                        mask_batch[k, j] = image_dict['mask']
                inputs = img_batch.to(device)
                masks = mask_batch.to(device)
                model.load_state_dict(new_state)

                # generate mask for each of the first elements in the batch
                with torch.set_grad_enabled(False):
                    outputs = model(inputs)
                    for j in range(int(outputs.shape[0])):
                        if j*batch_size >= int(len(img)):
                            break
                        mask_img = outputs[j][0].data.cpu().numpy()
                        mask_img = 1.0 / (1.0 + np.exp(-mask_img))
                        retransform = Resize((854, 480), (854, 480))
                        mask_img = retransform({'image': mask_img, 'mask': mask_img})
                        mask_img = np.asarray(mask_img['image'] > 0.5, dtype=np.uint8)
                        img_png = img[j*batch_size][:-4] + '.png'
                        print(img_png)
                        cv2.imwrite(os.path.join(target_dir, seq_name, img_png), mask_img)
                        #plt.subplot(1, 1, 1)
                        #plt.imshow(mask_img)
                #plt.show(block=True)

                # classify batch with refined model
                loss_refined, score_refined = classify_baseline_sequence(inputs, masks, model, criterion, metrics)
                # classify batch with unrefined model (comparison)
                model.load_state_dict(initial_state)
                loss_unrefined, score_unrefined = classify_baseline_sequence(inputs, masks, model, criterion, metrics)
                # log to tensorboard
                writer.add_scalars('Accuracy/f1_score', {'refined': score_refined,
                                                         'unrefined': score_unrefined,
                                                         'gain': score_refined - score_unrefined}, n_iter)
                writer.add_scalars('Loss', {'refined': loss_refined,
                                            'unrefined': loss_unrefined,
                                            'gain': loss_refined - loss_unrefined}, n_iter)
    writer.close()
    model.load_state_dict(initial_state)


def standard_online_loop(inputs, masks, iterations, model, criterion, optimizer):
    writer = SummaryWriter('online_plots')
    for i in range(iterations):
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)[:, 0]
            loss = criterion(outputs.unsqueeze(1), masks[:, 0])
            print('online loss:' + str(loss))
            loss.backward()
            optimizer.step()
            writer.add_scalars('Loss', {'loss': loss}, i)
    writer.close()
    return model.state_dict()


def classify_sequence(frames, masks, model, criterion, metrics):
    with torch.set_grad_enabled(False):
        outputs = model(frames)
        loss = criterion(outputs.unsqueeze(2), masks)
        y_pred = 1.0 / (1 + np.exp(-outputs.data.cpu().numpy().ravel()))
        y_true = masks.data.cpu().numpy().ravel()
        for name, metric in metrics.items():
            if name == 'f1_score':
                score = metric(y_true > 0, y_pred > 0.5)
    return loss, score


def repeat_oneshot_validation(model, criterion, optimizer, metrics, root_dir, target_dir, iterations, imageFolder,
                       maskFolder, img_size, batch_size, step):
    writer = SummaryWriter('drive/My Drive/oneshot/runs/TAGNN_SGD_1e-5_2iter')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    n_iter = 0
    # cache initial model weights
    initial_state = copy.deepcopy(model.state_dict())

    transform = transforms.Compose([Resize(img_size, img_size), ToTensor(), Normalize()])
    info_file = open(root_dir + 'db_info.yaml', 'r')
    seq_list = yaml.load(info_file)['sequences']
    for seq in seq_list:
        if seq['year'] == 2016 and seq['set'] == 'val':
            seq_name = seq['name']
            print(seq_name)
            os.mkdir(os.path.join(target_dir, seq_name))
            image_names = sorted(glob.glob(os.path.join(root_dir, imageFolder, seq_name, '*')))
            new_image_names = []
            mask_names = sorted(glob.glob(os.path.join(root_dir, maskFolder, seq_name, '*')))
            new_mask_names = []

            # create sequence dataset {{x, x+10, x+20},{x+1, x+1+10, x+1+20},...}
            for elem in range(len(image_names)):
                if elem + step * (batch_size - 1) < len(image_names):
                    for frame in range(batch_size):
                        new_image_names.append(image_names[elem + frame * step])
                        new_mask_names.append(mask_names[elem + frame*step])
                else:
                    for frame in range(batch_size):
                        new_image_names.append(image_names[elem + frame * step - len(image_names)])
                        new_mask_names.append(mask_names[elem + frame * step - len(image_names)])
            image_names = new_image_names
            mask_names = new_mask_names

            # iterate through sequence
            for i in range(int(len(image_names) / (batch_size*batch_size)) + 1):
                n_iter += 1
                img = []

                # create batch
                img_batch = torch.zeros(3, batch_size, 3, img_size[1], img_size[0])
                mask_batch = torch.zeros(3, batch_size, 1, img_size[1], img_size[0])
                for k in range(img_batch.shape[0]):
                    for j in range(batch_size):
                        if i * batch_size*batch_size + k * batch_size  + j >= int(len(image_names)):
                            break
                        img.append(image_names[i * batch_size*batch_size + k*batch_size + j][-9:])
                        image = cv2.imread(image_names[i * batch_size*batch_size + k*batch_size + j], 1).transpose(2, 0, 1)
                        mask = cv2.imread(mask_names[i * batch_size*batch_size + k*batch_size + j], 0)
                        mask = np.asarray(mask > 0).astype(float)
                        image_dict = {'image': image, 'mask': mask}
                        image_dict = transform(image_dict)
                        img_batch[k, j] = image_dict['image']
                        mask_batch[k, j] = image_dict['mask']
                # change every last element to the first frame
                for j in range(batch_size):
                    image = cv2.imread(image_names[0], 1).transpose(2, 0, 1)
                    mask = cv2.imread(mask_names[0], 0)
                    mask = np.asarray(mask > 0).astype(float)
                    image_dict = {'image': image, 'mask': mask}
                    image_dict = transform(image_dict)
                    img_batch[j, 2] = image_dict['image']
                    mask_batch[j, 2] = image_dict['mask']
                inputs = img_batch.to(device)
                masks = mask_batch.to(device)

                # do online training
                model.train()
                model.load_state_dict(initial_state)
                new_model_state = repeat_online_loop(inputs, masks, iterations, model, criterion, optimizer)
                model.load_state_dict(new_model_state)
                new_state = copy.deepcopy(model.state_dict())
                model.load_state_dict(new_state)

                # generate output masks for first frame for every element in the batch
                model.eval()
                with torch.set_grad_enabled(False):
                    outputs = model(inputs)
                    for j in range(int(outputs.shape[0])):
                        if j*batch_size >= int(len(img)):
                            break
                        mask_img = outputs[j][0].data.cpu().numpy()
                        mask_img = 1.0 / (1.0 + np.exp(-mask_img))
                        retransform = Resize((854, 480), (854, 480))
                        mask_img = retransform({'image': mask_img, 'mask': mask_img})
                        mask_img = np.asarray(mask_img['image'] > 0.5, dtype=np.uint8)
                        img_png = img[j*batch_size][:-4] + '.png'
                        print(img_png)
                        cv2.imwrite(os.path.join(target_dir, seq_name, img_png), mask_img)
                        #plt.subplot(1, 1, 1)
                        #plt.imshow(mask_img)
                #plt.show(block=True)

                # classify batch with refined model
                loss_refined, score_refined = classify_baseline_sequence(inputs, masks, model, criterion, metrics)
                # classify batch with unrefined model (comparison)
                model.load_state_dict(initial_state)
                loss_unrefined, score_unrefined = classify_baseline_sequence(inputs, masks, model, criterion, metrics)
                # log to tensorboard
                writer.add_scalars('Accuracy/f1_score', {'refined': score_refined,
                                                         'unrefined': score_unrefined,
                                                         'gain': score_refined - score_unrefined}, n_iter)
                writer.add_scalars('Loss', {'refined': loss_refined,
                                            'unrefined': loss_unrefined,
                                            'gain': loss_refined - loss_unrefined}, n_iter)
    writer.close()
    model.load_state_dict(initial_state)


def repeat_online_loop(inputs, masks, iterations, model, criterion, optimizer):
    writer = SummaryWriter('online_plots')
    for i in range(iterations):
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)[:, 2]
            loss = criterion(outputs.unsqueeze(1), masks[:, 2])
            print('online loss:' + str(loss))
            loss.backward()
            optimizer.step()
            writer.add_scalars('Loss', {'loss': loss}, i)
    writer.close()
    return model.state_dict()



def past_oneshot_validation(model, criterion, optimizer, metrics, root_dir, target_dir, iterations, imageFolder,
                       maskFolder, img_size, batch_size, step):
    writer = SummaryWriter('drive/My Drive/oneshot/runs/TAGNN_SGD_1e-5_2iter')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    n_iter = 0
    # cache initial model weights
    initial_state = copy.deepcopy(model.state_dict())

    transform = transforms.Compose([Resize(img_size, img_size), ToTensor(), Normalize()])
    info_file = open(root_dir + 'db_info.yaml', 'r')
    seq_list = yaml.load(info_file)['sequences']
    for seq in seq_list:
        if seq['year'] == 2016 and seq['set'] == 'val':
            seq_name = seq['name']
            print(seq_name)
            os.mkdir(os.path.join(target_dir, seq_name))
            image_names = sorted(glob.glob(os.path.join(root_dir, imageFolder, seq_name, '*')))
            new_image_names = []
            mask_names = sorted(glob.glob(os.path.join(root_dir, maskFolder, seq_name, '*')))
            new_mask_names = []

            # create sequence dataset {{x, x+10, x+20},{x+1, x+1+10, x+1+20},...}
            for elem in range(len(image_names)):
                if elem + step * (batch_size - 1) < len(image_names):
                    for frame in range(batch_size):
                        new_image_names.append(image_names[elem + frame * step])
                        new_mask_names.append(mask_names[elem + frame * step])
                else:
                    for frame in range(batch_size):
                        new_image_names.append(image_names[elem + frame * step - len(image_names)])
                        new_mask_names.append(mask_names[elem + frame * step - len(image_names)])
            image_names = new_image_names
            mask_names = new_mask_names

            # iterate through sequence
            for i in range(int(len(image_names) / (batch_size * batch_size)) + 1):
                n_iter += 1
                img = []

                # create batch
                img_batch = torch.zeros(3, batch_size, 3, img_size[1], img_size[0])
                mask_batch = torch.zeros(3, batch_size, 1, img_size[1], img_size[0])
                for k in range(img_batch.shape[0]):
                    for j in range(batch_size):
                        if i * batch_size * batch_size + k * batch_size + j >= int(len(image_names)):
                            break
                        img.append(image_names[i * batch_size * batch_size + k * batch_size + j][-9:])
                        image = cv2.imread(image_names[i * batch_size * batch_size + k * batch_size + j], 1).transpose(
                            2, 0, 1)
                        mask = cv2.imread(mask_names[i * batch_size * batch_size + k * batch_size + j], 0)
                        mask = np.asarray(mask > 0).astype(float)
                        image_dict = {'image': image, 'mask': mask}
                        image_dict = transform(image_dict)
                        img_batch[k, j] = image_dict['image']
                        mask_batch[k, j] = image_dict['mask']
                # change every last element to the first frame
                for j in range(batch_size):
                    image = cv2.imread(image_names[0], 1).transpose(2, 0, 1)
                    mask = cv2.imread(mask_names[0], 0)
                    mask = np.asarray(mask > 0).astype(float)
                    image_dict = {'image': image, 'mask': mask}
                    image_dict = transform(image_dict)
                    img_batch[j, 2] = image_dict['image']
                    mask_batch[j, 2] = image_dict['mask']
                # for every batch except the first change the middle element to the past frame and its label to its
                # predicted mask
                if i != 0:
                    for j in range(batch_size):
                        image = cv2.imread(
                            image_names[i * batch_size * batch_size + j * batch_size - batch_size * batch_size],
                            1).transpose(2, 0, 1)
                        image_dict = {'image': image, 'mask': image}
                        image_dict = transform(image_dict)
                        img_batch[j, 1] = image_dict['image']
                        mask_batch[j, 1] = input_masks[j]
                else:
                    for j in range(batch_size):
                        image = cv2.imread(image_names[0], 1).transpose(2, 0, 1)
                        mask = cv2.imread(mask_names[0], 0)
                        mask = np.asarray(mask > 0).astype(float)
                        image_dict = {'image': image, 'mask': mask}
                        image_dict = transform(image_dict)
                        img_batch[j, 1] = image_dict['image']
                        mask_batch[j, 1] = image_dict['mask']
                inputs = img_batch.to(device)
                masks = mask_batch.to(device)

                # do online training
                model.train()
                model.load_state_dict(initial_state)
                new_model_state = past_online_loop(inputs, masks, iterations, model, criterion, optimizer)
                model.load_state_dict(new_model_state)
                new_state = copy.deepcopy(model.state_dict())
                model.load_state_dict(new_state)

                #save past masks
                input_masks = torch.zeros(3, 1, img_size[1], img_size[0])

                # generate output masks for first frame for every element in the batch
                model.eval()
                with torch.set_grad_enabled(False):
                    outputs = model(inputs)
                    for j in range(int(outputs.shape[0])):
                        if j * batch_size >= int(len(img)):
                            break
                        input_masks[j] = 1.0 / (1.0 + torch.exp(-outputs[j][1].unsqueeze(0))) > 0.5
                        mask_img = outputs[j][0].data.cpu().numpy()
                        mask_img = 1.0 / (1.0 + np.exp(-mask_img))
                        retransform = Resize((854, 480), (854, 480))
                        mask_img = retransform({'image': mask_img, 'mask': mask_img})
                        mask_img = np.asarray(mask_img['image'] > 0.5, dtype=np.uint8)
                        img_png = img[j * batch_size][:-4] + '.png'
                        print(img_png)
                        cv2.imwrite(os.path.join(target_dir, seq_name, img_png), mask_img)
                        #plt.subplot(1, 1, 1)
                        #plt.imshow(mask_img)
                #plt.show(block=True)

                # classify batch with refined model
                loss_refined, score_refined = classify_baseline_sequence(inputs, masks, model, criterion, metrics)
                # classify batch with unrefined model (comparison)
                model.load_state_dict(initial_state)
                loss_unrefined, score_unrefined = classify_baseline_sequence(inputs, masks, model, criterion, metrics)
                # log to tensorboard
                writer.add_scalars('Accuracy/f1_score', {'refined': score_refined,
                                                         'unrefined': score_unrefined,
                                                         'gain': score_refined - score_unrefined}, n_iter)
                writer.add_scalars('Loss', {'refined': loss_refined,
                                            'unrefined': loss_unrefined,
                                            'gain': loss_refined - loss_unrefined}, n_iter)
    writer.close()
    model.load_state_dict(initial_state)


def past_online_loop(inputs, masks, iterations, model, criterion, optimizer):
    writer = SummaryWriter('online_plots')
    for i in range(iterations):
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)[:, 1:3]
            loss = criterion(outputs.unsqueeze(2), masks[:, 1:3])
            print('online loss:' + str(loss))
            loss.backward()
            optimizer.step()
            writer.add_scalars('Loss', {'loss': loss}, i)
    writer.close()
    return model.state_dict()


def optimize_attention(image, mask, model, seq):
    model.eval()
    B, F, C, H, W = image.shape
    mask = mask.squeeze()
    out = model.get_features(image, mask)
    mask = out['mask'][seq]
    X = out['feature'][seq]
    N, C = X.shape
    X_X = torch.mm(X, X.T)
    X_X = X_X[mask > 0]
    X_X = X_X[:, mask > 0]
    N, _ = X_X.shape
    # X_X[XX_mask<=1] *= 0     # similarity between feature in and out mask should be low
    # X_X[XX_mask==0] = 0       # similarity between background features should be neglegted
    W = torch.mm(torch.mm(torch.ones(C, N).cuda(), X_X), torch.ones(N, C).cuda())
    W = W / torch.sum(W)
    return W