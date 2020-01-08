import time
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt

def train_model(model, criterion, dataloader, optimizer, metrics, num_epochs=1):
  since = time.time()
  writer = SummaryWriter()
  best_loss = 1e10
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.to(device)
  n_iter = 0
  fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Test_{m}' for m in metrics.keys()]

  for epoch in range(1, num_epochs+1):
    print('Epoch {}/{}'.format(epoch, num_epochs))
    print('-' * 10)

    batchsummary = {a: [0] for a in fieldnames}

    for phase in ['Train']:
      if phase == 'Train':
          model.train()  # Set model to training mode
      else:
          model.eval()   # Set model to evaluate mode
      for sample in tqdm(iter(dataloader)):
        n_iter += 1
        inputs = sample['image'].to(device)
        masks = sample['mask'].to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(phase == 'Train'):
          outputs = model(inputs)
          loss = criterion(outputs['out'], masks)
          y_pred = 1.0/(1+np.exp(-outputs['out'].data.cpu().numpy().ravel()))
          y_true = masks.data.cpu().numpy().ravel()
          for name, metric in metrics.items():
            if name == 'f1_score':
              score = metric(y_true > 0, y_pred > 0.5)
              batchsummary[f'{phase}_{name}'].append(score)
              writer.add_scalar('Accuracy/f1_score',score,n_iter)
          if phase == 'Train':
            loss.backward()
            optimizer.step()
          writer.add_scalar('Loss/Train',loss,n_iter)
          best_loss = loss if loss < best_loss else best_loss
      batchsummary['epoch'] = epoch
      epoch_loss = loss
      batchsummary[f'{phase}_loss'] = epoch_loss.item()
      print('{} Loss: {:.4f}'.format(phase,loss))
    for field in fieldnames[3:]:
      batchsummary[field] = np.mean(batchsummary[field])
    print(batchsummary)

    #grid = make_grid(outputs['out'])
    #writer.add_image('masks', grid, epoch)

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(
      time_elapsed // 60, time_elapsed % 60))
  print('Lowest Loss: {:4f}'.format(best_loss))
  writer.close()
  return model

def train_full_model(model, criterion, dataloader, optimizer, metrics, num_epochs=1):
  since = time.time()
  writer = SummaryWriter()
  best_loss = 1e10
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.to(device)
  n_iter = 0
  fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Test_{m}' for m in metrics.keys()]

  for epoch in range(1, num_epochs+1):
    print('Epoch {}/{}'.format(epoch, num_epochs))
    print('-' * 10)

    batchsummary = {a: [0] for a in fieldnames}

    for phase in ['Train']:
      if phase == 'Train':
          model.train()  # Set model to training mode
      else:
          model.eval()   # Set model to evaluate mode
      for sample in tqdm(iter(dataloader)):
        n_iter += 1
        inputs = sample['image'].to(device)
        masks = sample['mask'].to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(phase == 'Train'):
          outputs = model(inputs)
          loss = criterion(outputs, masks)
          y_pred = 1.0/(1+np.exp(-outputs.data.cpu().numpy().ravel()))
          y_true = masks.data.cpu().numpy().ravel()
          for name, metric in metrics.items():
            if name == 'f1_score':
              score = metric(y_true > 0, y_pred > 0.5)
              batchsummary[f'{phase}_{name}'].append(score)
              writer.add_scalar('Accuracy/f1_score',score,n_iter)
          if phase == 'Train':
            loss.backward()
            optimizer.step()
          writer.add_scalar('Loss/Train',loss,n_iter)
          best_loss = loss if loss < best_loss else best_loss
      batchsummary['epoch'] = epoch
      epoch_loss = loss
      batchsummary[f'{phase}_loss'] = epoch_loss.item()
      print('{} Loss: {:.4f}'.format(phase,loss))
    for field in fieldnames[3:]:
      batchsummary[field] = np.mean(batchsummary[field])
    print(batchsummary)

    #grid = make_grid(outputs['out'])
    #writer.add_image('masks', grid, epoch)

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(
      time_elapsed // 60, time_elapsed % 60))
  print('Lowest Loss: {:4f}'.format(best_loss))
  writer.close()
  return model

def show_results(model, dataloader, number):
  model.eval()
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  counter = 0
  max_count = int(number/dataloader.batch_size)
  for sample in dataloader:
    inputs = sample['image'].to(device)
    masks = sample['mask'].to(device)
    with torch.set_grad_enabled(False):
      outputs = model(inputs)
      y_pred = outputs['out'].data.cpu().numpy()
      y_true = masks.data.cpu().numpy()
      images = inputs.cpu().numpy()
      for i in range(dataloader.batch_size):
        image = images[i]
        mask_pred = y_pred[i][0]
        mask = y_true[i][0]
        print(np.max(image[1,:,:]), np.min(mask_pred))
        print(np.max(mask), np.min(mask))
        print(image.shape,mask_pred.shape,mask.shape)
        plt.subplot(3,1,1)
        plt.imshow(image.transpose(1,2,0))
        plt.subplot(3,1,2)
        mask_pred = 1.0/(1+np.exp(-mask_pred))
        plt.imshow(mask_pred>0.5)
        plt.subplot(3,1,3)
        plt.imshow(mask>0)
        plt.show(block=True)
      counter += 1
      if counter >= max_count:
        return

def show_results_img(model, dataloader, number):
  model.eval()
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  counter = 0
  max_count = int(number/dataloader.batch_size)
  for sample in dataloader:
    inputs = sample['image'].to(device)
    masks = sample['mask'].to(device)
    with torch.set_grad_enabled(False):
      outputs = model(inputs)
      y_pred = outputs['out'].data.cpu().numpy()
      y_true = masks.data.cpu().numpy()
      images = inputs.cpu().numpy()
      for i in range(dataloader.batch_size):
        image = images[i]
        mask_pred = y_pred[i][0]
        mask = y_true[i][:]
        mask = np.squeeze(np.sum(mask, axis=0))
        gt_img = image + np.asarray([mask, -mask, -mask],dtype=float)*0.3
        image += np.asarray([1/(1+np.exp(-mask_pred))>0.5, -1*(1/(1+np.exp(-mask_pred))>0.5), -1*(1/(1+np.exp(-mask_pred))>0.5)],dtype=float)*0.2
        #print(np.max(image[0,:,:]), np.min(mask_pred))
        #print(np.max(mask), np.min(mask))
        print(image.shape,mask_pred.shape,mask.shape)
        plt.subplot(2,1,1)
        plt.imshow(image.transpose(1,2,0))
        plt.subplot(2,1,2)
        #mask_pred = 1.0/(1+np.exp(-mask_pred))
        plt.imshow(gt_img.transpose(1,2,0))
        #plt.subplot(3,1,3)
        #plt.imshow(mask>0)
        plt.show(block=True)
      counter += 1
      if counter >= max_count:
        return