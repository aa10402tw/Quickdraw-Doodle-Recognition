import numpy as np 
import pandas as pd 
from tqdm import tqdm_notebook as tqdm
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw 
from dask import bag
from glob import glob

import cv2
import time
import os
import re
import ast
import copy
import argparse

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
from torch.optim import lr_scheduler
from torch.autograd import Variable

import numpy as np 
import pandas as pd 
from tqdm import tqdm_notebook as tqdm
from glob import glob
import os

classes_path = os.listdir('train_simplified/')
classes_path = sorted(classes_path, key=lambda s: s.lower())
labels = [x[:-4].replace(" ", "_") for i, x in enumerate(classes_path)]

indexes = [i for i in range(len(labels))]
mapping_l2c = dict(zip(labels, indexes))
mapping_c2l = dict(zip(indexes, labels))

print(len(mapping_l2c))
def label_to_categorical(label):
    label = label.replace(' ', '_')
    return mapping_l2c[label]

def categorical_to_label(category):
    return mapping_c2l[category]

BASE_SIZE = 256
size = 256

def draw_cv2(raw_strokes, size=256, lw=6, time_color=True):
    raw_strokes = eval(raw_strokes)
    img = np.zeros((2, size, size), np.uint8)
    
    # origin info with time info
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img[0], (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    # First half
    length = len(raw_strokes)
    for t, stroke in enumerate(raw_strokes[:(length+1)//2]):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 26 if time_color else 255
            _ = cv2.line(img[1], (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw//2)
#     # Second half
#     for t, stroke in enumerate(raw_strokes[(length+1)//2:]):
#         for i in range(len(stroke[0]) - 1):
#             color = 255 - min(t, 10) * 26 if time_color else 255
#             _ = cv2.line(img[2], (stroke[0][i], stroke[1][i]),
#                          (stroke[0][i + 1], stroke[1][i + 1]), color, lw//2)
    
#     # Before last
#     for t, stroke in enumerate(raw_strokes[:(length-1)]):
#         tmp_img = np.zeros((size, size), np.uint8)
#         for i in range(len(stroke[0]) - 1):
#             color = 64
#             _ = cv2.line(tmp_img, (stroke[0][i], stroke[1][i]),
#                          (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
#             img[3] += tmp_img
        

#     # After first
#     for t, stroke in enumerate(raw_strokes[1:]):
#         tmp_img = np.zeros((size, size), np.uint8)
#         for i in range(len(stroke[0]) - 1):
#             color = 64
#             _ = cv2.line(tmp_img, (stroke[0][i], stroke[1][i]),
#                          (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
#             img[4] += tmp_img
            
#     img[img > 255] = 255
            
    if size != 256:
        img = np.rollaxis(img, (0, 3))
        img = cv2.resize(img, (size, size))
        return np.rollaxis(img, (2, 1))
    else:
        return img

all_csv = glob('train_simplified/*.csv')
print(len(all_csv))

def df_chunk_train(csv_file, chunksize):
    while True:
        for df in pd.read_csv(csv_file, chunksize=10240, usecols=['drawing', 'word']):
            yield df
            
def df_chunk_val(csv_file, chunksize):
    while True:
        for df in pd.read_csv(csv_file, chunksize=34000, usecols=['drawing', 'word']):

            yield df

class DrawDataset(torch.utils.data.Dataset):
    def __init__(self, size=256, train=True):
        self.size = size
        self.remain = 0
        self.df = None
        self.idx = 0
        self.train = train
        if train:
            t = time.time()
            self.df = pd.read_csv('train_data/train.csv')
            print('Read Csv take', time.time()-t, 'sec')
        else:
            self.df = pd.read_csv('train_data/val.csv')
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        raw_stroke = self.df['drawing'].values[idx]
        x = draw_cv2(raw_stroke, size=size) / 255.0
        y = label_to_categorical(self.df['word'].values[idx].replace(' ', '_'))
        return (x, y)

all_csv = glob('train_simplified/*.csv')
REAL_BATCH_SIZE = 1024
BATCH_SIZE = 64
batch_size = BATCH_SIZE

train_Dataset = DrawDataset(size=256, train=True)
train_loader = DataLoader(train_Dataset, batch_size=batch_size, shuffle=True)

val_Dataset = DrawDataset(size=256, train=False)
val_loader = DataLoader(val_Dataset, batch_size=batch_size)

plt.figure(figsize=(20, 20))
for i, (x, y) in enumerate(train_loader):
    plt.subplot(5, 5, i+1)
    plt.imshow(x[0].permute(1,2,0)[:,:,0])
    plt.title(categorical_to_label(y.data.numpy()[0]))
    if i >= 24:
        print(x.shape)
        print(y.shape)
        break
plt.show()

# from ResNeXt import*
from ResNeXt_v2 import*
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def save_net(net, path='trained_model/ResNeXt_v2.pkl'):
    torch.save(net.state_dict(), path)
    
def load_net(path=None):
    net = ResNeXt29_4x64d(num_classes=340)
    net.cuda()
    net = torch.nn.DataParallel(net)
    torch.backends.cudnn.benchmark = True
    if path is not None:
        net.load_state_dict(torch.load(path))
    return net

class Scheduler:
    def __init__(self, k=7):
        self.k = k
        self.tolerance = 0
        self.best_loss = 100
        
    def no_patience(self, cur_loss):
        if cur_loss < self.best_loss:
            self.best_loss = cur_loss
            self.tolerance = 0
            return False
        else:
            self.tolerance += 1
            if self.tolerance >= self.k:
                self.tolerance = 0
                return True
            else:
                return False
    
def topk_acc(outputs, y, topk=(1,3)):
    maxk = max(topk)
    target = y
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True) 
        res.append(correct_k.mul_(100.0 / 100.0))
    top_1, top_3 = res
    correct_top1 = top_1.item()
    correct_top3 = top_3.item()
    return correct_top1, correct_top3

def eval_model(net, val_loader, USE_GPU=True):
    net.eval()
    
    sum_loss, total, correct_top1, correct_top3 = 0, 0, 0, 0
    criterion = nn.CrossEntropyLoss()
    
    pbar = tqdm(total=len(val_loader), unit=' iters')
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            x, y = data
            if USE_GPU:
                device = 'cuda'
            else:
                device = 'cpu'
            x = x.to(device=device, dtype=torch.float)
            y = y.to(device=device, dtype=torch.long)
            outputs = net(x)
            loss    = criterion(outputs, y)
            c_top1, c_top3 = topk_acc(outputs, y, topk=(1,3))

            sum_loss += loss.item()
            correct_top1 += c_top1
            correct_top3 += c_top3

            batch_size = int(y.data.cpu().size(0))
            total += batch_size
            pbar.set_postfix( {'loss': '%.4f'%(sum_loss/(i+1)), 'top1_acc': '%.4f'%(correct_top1/total), 
                                   'top3_acc': '%.4f'%(correct_top3/total) } )
            pbar.update()
    pbar.close()
    return sum_loss/(i+1), correct_top1/total, correct_top3/total

def train_model(net, train_loader, criterion, optimizer, num_batch_to_update=2, num_real_batch_to_eval=1000, start_iter=0, USE_GPU=True, save_best=True): 

    global LR
    best_val_loss = 100
    #num_batch_to_update = REAL_BATCH_SIZE//BATCH_SIZE
    print('Number of Totla Iters:', len(train_loader))
    print('batch_size =', BATCH_SIZE)
    print('num_accu =', num_batch_to_update)
    print('accu_batch_size = ', BATCH_SIZE*num_batch_to_update)
    print('num_batch_to_eval = ', num_real_batch_to_eval)
    optimizer.zero_grad() 
    total_loss = 0
    num_real_batch = 0
    total, correct_top1, correct_top3 = 0, 0, 0
    
    pbar = tqdm(total = num_real_batch_to_eval, unit=' batches')
    pbar.set_description('(Training)')
    net.train()
    
    
    lr_scheduler = Scheduler(k=1)
    t = time.time()
    for i, data in enumerate(train_loader):
        
        if(num_real_batch != 0 and num_real_batch % num_real_batch_to_eval == 0):
            pbar.close()
            time.sleep(1)
            print('eval')
            loss, top1_acc, top3_acc = eval_model(net, val_loader)
            
            if lr_scheduler.no_patience(loss):
                print('New LR:')
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] * 0.5
                print(g['lr'])
            
            total_loss = 0
            num_real_batch = 0
            total, correct_top1, correct_top3 = 0, 0, 0
            
            pbar = tqdm(total = num_real_batch_to_eval, unit=' batches')
            pbar.set_description('(Training (iter:%d)'%(i))
            net.train()
        
        x, y = data
        if USE_GPU:
            device = 'cuda'
        else:
            device = 'cpu'
        x = x.to(device=device, dtype=torch.float)
        y = y.to(device=device, dtype=torch.long)
        
        # forward
        outputs = net(x)
        loss    = criterion(outputs, y)
        loss.backward()
        total_loss += loss.item()
        
        # Compute correct
        batch_size = int(y.data.cpu().size(0))
        total += batch_size
        _, predicted = torch.max(outputs.data.cpu(), 1)
        
        c_top1, c_top3 = topk_acc(outputs, y, topk=(1,3))
        correct_top1 += c_top1
        correct_top3 += c_top3
                
        if (i+1) % num_batch_to_update == 0:
            optimizer.step()
            optimizer.zero_grad() 
            num_real_batch += 1
            pbar.set_postfix( {'loss': '%.4f'%(total_loss/(total//batch_size)), 'top1_acc': '%.4f'%(correct_top1/total), 
                                   'top3_acc': '%.4f'%(correct_top3/total) } )
            pbar.update()
                

# net = load_net()
net = load_net('trained_model/ResNeXt_v2.pkl')
# net = load_net(path='trained_model/ResNeXt.pkl') # now test acc :0.921 (train top3=0.9560)

LR = 0.001
# LR = 0.005 

# # num_batch_to_update = 256//BATCH_SIZE
# # num_batch_to_update = 512//BATCH_SIZE
# # num_batch_to_update = 288//BATCH_SIZE
# num_batch_to_update = 480//BATCH_SIZE
# num_batch_to_update = 960//BATCH_SIZE
# num_batch_to_update = 1920//BATCH_SIZE
num_batch_to_update = 3840//BATCH_SIZE

optimizer = optim.Adam(net.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
# criterion = topk_crossEntrophy(top_k=0.5) # Online Hard Example Mining


train_model(net, train_loader, criterion, optimizer, num_batch_to_update=num_batch_to_update, num_real_batch_to_eval=256)

save_net(net, path='trained_model/ResNeXt_v2.pkl')
# save_net(net, path='trained_model/ResNeXt.pkl')

# test on Train set
net = load_net(path='trained_model/ResNeXt_v2.pkl')
net.eval()
plt.figure(figsize=(20, 25))
for i, (x, y) in enumerate(train_loader):
    plt.subplot(5, 5, i+1)
    plt.imshow(x[0][0], cmap='gray')
    with torch.no_grad():
        x = x.to(device='cuda', dtype=torch.float)
        outputs = net(x)
        topk = (1, 3)
        maxk = max(topk)
        _, y_pred = outputs.topk(maxk, 1, True, True)
    plt.title('[GT.{}]'.format(categorical_to_label(y.data.numpy()[0])) + '\n' + 
              'P1.{}'.format(categorical_to_label(y_pred.cpu().data.numpy()[0][0])) + '\n' + 
              'P2.{}'.format(categorical_to_label(y_pred.cpu().data.numpy()[0][1])) + '\n' + 
              'P3.{}'.format(categorical_to_label(y_pred.cpu().data.numpy()[0][2]))
             )
    if i >= 24:
        print(x.shape)
        break
plt.show()