from __future__ import division
import os
import random
import numpy as np
import cv2
from torch.utils import data
import sys
import matplotlib.pyplot as plt

sys.path.append("..")
from datasets.utils import *

class my_dataset(data.Dataset):
    def __init__(self, split, aug=True, dataset = 'my'):
        self.intrinsics_color = np.array([[636.7395, 0.0, 630.6644],
                       [0.0,     636.1763, 366.0309],
                       [0.0,     0.0,  1.0]])
        self.intrinsics_color_inv = np.linalg.inv(self.intrinsics_color)            # 颜色相机内参的逆
        self.split = split
        self.data = ['seq1','seq2','seq3'] if self.split == 'train' else ['seq3']
        self.aug = aug
        self.dataset = dataset
        self.frame = []
        self.RT = []
        for i in self.data:
            with open('/mnt/sda2/xt/D455/' + i + '/rgb.txt') as f:
                self.frames = f.readlines()
                for j in self.frames:
                    self.frame.append((i, j.split(' ')[1][4:-5]))
        #print(self.frame)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        seq_id, id = self.frame[index]

        objs = {}
        objs['color'] = '/mnt/sda2/xt/D455/'+ seq_id + '/rgb' + '/' + id + '.png'
        objs['depth'] = '/mnt/sda2/xt/D455/'+ seq_id + '/depth' + '/' + id + '.png'        # Twc
        objs['pose'] = '/mnt/sda2/xt/D455/'+ seq_id + '/pose' + '/' + id + '.npy'

        img = cv2.imread(objs['color'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.resize(img, (640, 480))
        # plt.imshow(img)
        # plt.show()
        pose = np.load(objs['pose'])
        
        if self.split == 'test':
            img, pose = to_tensor_query(img, pose)
            return img, pose
        pose[0:3,3] = pose[0:3,3] * 1000
        depth = cv2.imread(objs['depth'],-1)
        #depth = cv2.resize(depth,(640,480))
        
        coord, mask = get_coord(depth, pose, self.intrinsics_color_inv, self.dataset)
        # plt.imshow(coord.sum(axis=2))
        # plt.show()

       
        img, coord, mask = data_aug(img, coord, mask, self.aug)
        
        coord = coord[4::8,4::8,:]  # [60 80]
        mask = mask[4::8,4::8].astype(np.float16)

        img, coord, mask  = to_tensor(img, coord, mask)
        #pose[0:3,3] = pose[0:3,3] / 1000
        #pose = torch.from_numpy(pose).float()
        
        return img, coord, mask
if __name__ == '__main__':
    datat = my_dataset(split='train')
    trainloader = data.DataLoader(datat, batch_size=1, num_workers=1, shuffle=True, drop_last = True)  #

    for _, (img, coord, mask) in enumerate(trainloader): 
        print(img.shape)
        print(coord.shape)
        print(mask.shape)

