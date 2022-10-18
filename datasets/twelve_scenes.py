from __future__ import division

import os
import random
import numpy as np
import cv2
from torch.utils import data
import sys

sys.path.append("..")
from datasets.utils import *


class TwelveScenes(data.Dataset):
    def __init__(self, root, dataset='12S', scene='apt2/bed', split='train', model='fdanet', aug='True'):
        self.intrinsics_color = np.array([[572.0, 0.0,     320.0],
                       [0.0,     572.0, 240.0],
                       [0.0,     0.0,  1.0]])

        self.intrinsics_color_inv = np.linalg.inv(self.intrinsics_color)
        
        self.model = model
        self.dataset = dataset
        self.aug = aug 
        self.root = os.path.join(root,'12Scenes')
        self.scene = scene
        
        self.split = split
        self.obj_suffixes = ['.color.jpg', '.pose.txt', '.depth.png', 
                '.label.png']
        self.obj_keys = ['color', 'pose', 'depth', 'label']

        if self.dataset == '12S' and self.split == 'test':
            with open(os.path.join(self.root, self.scene, 
                    '{}{}'.format(self.split, '.txt')), 'r') as f:
                self.frames = f.readlines()
        else:
            self.frames = []
            for scene in self.scenes:
                with open(os.path.join(self.root, scene, 
                        '{}{}'.format(self.split, '.txt')), 'r') as f:
                    frames = f.readlines()
                    frames = [scene + ' ' + frame for frame in frames ]
                self.frames.extend(frames)


    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        
        frame = self.frames[index].rstrip('\n')
        seq_id, frame_id = frame.split('-')

        objs = {}
        

        # 加载数据集中的文件
        objs['color'] = '/home/k545/sda1/xietao/12Scenes/' + self.scene + '/data/' + seq_id + '-' + frame_id + '.color.jpg'
        objs['depth'] = '/home/k545/sda1/xietao/12Scenes/' + self.scene + '/data/' + seq_id + '-' + frame_id + '.depth.png'        # Twc
        objs['pose'] = '/home/k545/sda1/xietao/12Scenes/' + self.scene + '/data/' + seq_id + '-' + frame_id + '.pose.txt'

        img = cv2.imread(objs['color'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480))

        pose = np.loadtxt(objs['pose'])
       
        if self.split == 'test':
            img, pose = to_tensor_query(img, pose)
            return img, pose
        
        depth = cv2.imread(objs['depth'],-1)
        
        pose[0:3,3] = pose[0:3,3] * 1000
    
        coord, mask = get_coord(depth, pose, self.intrinsics_color_inv, self.dataset)
        
        img, coord, mask = data_aug(img, coord, mask, self.aug)
        
        if self.model == 'hscnet':
            coord = coord
        coord = coord[4::8,4::8,:]  # [60 80]
        mask = mask[4::8,4::8].astype(np.float16)
        img, coord, mask  = to_tensor(img, coord, mask)
        return img, coord, mask


if __name__ == '__main__':
    dataset = TwelveScenes(root="/home/k545/sda1/xietao/12Scenes/")
    trainloader = data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True, drop_last = True)  #

    for _, (img, coord, mask) in enumerate(trainloader): 
        print(img.shape)
        print(coord.shape)
        print(mask.shape)
        printt()
