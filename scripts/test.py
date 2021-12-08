from torch.utils import data
import sys
import os
import random
import argparse
from pathlib import Path
import torch
import time
from torch.autograd import Variable
import numpy as np 
import cv2


sys.path.insert(0, './pnpransac')
from pnpransac import pnpransac
from datasets import get_dataset
from models import get_model
from loss import *
from scripts.utils import get_pose_err

def test(args):
    if args.dataset == 'my':
        dataset = get_dataset('my')
        test_dataset = dataset(split = 'test', aug = False)
    else:
        if args.dataset == '7S':
            dataset = get_dataset('7S')  
        if args.dataset == '12S':
            dataset = get_dataset('12S')
        test_dataset = dataset(args.data_path, args.dataset, args.scene, split='test', model=args.model, aug='False')

    testloader = data.DataLoader(test_dataset, batch_size=1, num_workers=16, shuffle=False)  

    intrinsics_color = test_dataset.intrinsics_color                                 # RGB相机内参

    pose_solver = pnpransac(intrinsics_color[0,0], intrinsics_color[1,1], intrinsics_color[0,2], intrinsics_color[1,2])

    
    # prepare model
    torch.set_grad_enabled(False)                       # 设为不求导
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args.model, args.dataset)
    model_state = torch.load(args.resume, 
                map_location=device)['model_state']
    model.load_state_dict(model_state)                  # 加载模型参数
    model.to(device)
    model.eval()

    # state = {'model_state': model.state_dict()}
    # torch.save(state, '/mnt/sda2/xt/7scenes/lap/chess_supervise_PFFM.pth')
    # printt()

    # start evaluation
    rot_err_list = []
    transl_err_list = []

    if args.dataset == 'my':
        x = np.linspace(4, 1280-4, 160)
        y = np.linspace(4, 720-4, 90)
    else:
        x = np.linspace(4,640-4, 80)
        y = np.linspace(4, 480-4, 60)
    xx, yy = np.meshgrid(x, y)      # [60 80]
    pcoord = np.concatenate((np.expand_dims(xx,axis=2), 
            np.expand_dims(yy,axis=2)), axis=2)         # [60 80 2]
    index = 1
    for _, (img, pose) in enumerate(testloader):
        img_cpu = img;
        img = img.to(device)
        if args.model == 'fdanet':
            if args.dataset == '7S':
                # 考虑坐标
                x = np.linspace(0, 480 - 1, 480)
                y = np.linspace(0, 640 - 1, 640)
                xx, yy = np.meshgrid(y, x)  # 坐标网格化[img_width img_height]
                
                xx = torch.from_numpy(xx) / 640  # [480 640]
                yy = torch.from_numpy(yy) / 480
                xx = torch.unsqueeze(xx, 0)  # [1 480 640]
                yy = torch.unsqueeze(yy, 0)
                img_coord = torch.cat((xx, yy), 0).unsqueeze(0)  # [2 480 640]->[1 2 480 640]
                img_coord = img_coord.float()
                img_coord = Variable(img_coord).cuda()
                img = torch.cat((img, img_coord), 1)
            with torch.no_grad():
                coord, uncertainty, supervise_coord, supervise_uncen = model(img)
            
            coord = np.transpose(coord.cpu().data.numpy()[0,:,:,:], (1,2,0))            # [3 60 80]->[60 80 3]
            uncertainty = np.transpose(uncertainty[0].cpu().data.numpy(), (1,2,0))         # [60 80 1]
        
        coord = np.concatenate([coord,uncertainty],axis=2)  # [60 80 4]
        coord = np.ascontiguousarray(coord)             # 预测出的世界坐标    [60 80 3]
        pcoord = np.ascontiguousarray(pcoord)           # 转换为连续的数组    像素坐标 [60 80 2]
       
        pcoord = pcoord.reshape(-1,2)                   # 像素坐标           [4800 2]  numpy
        coords = coord[:,:,0:3].reshape(-1,3)           # 预测出的世界坐标     [4800 3] numpy
        confidences = coord[:,:,3].flatten().tolist()   # 预测出的置信度       [4800]数组
        
        coords_filtered = []        # 存放置信度高的世界坐标
        coords_filtered_2D = []     # 存放置信度高的像素坐标
        for i in range(len(confidences)):
            if confidences[i]>0:
                coords_filtered.append(coords[i])
                coords_filtered_2D.append(pcoord[i])
        

        coords_filtered = np.vstack(coords_filtered)                # [N 3] numpy
        coords_filtered_2D = np.vstack(coords_filtered_2D)          # [N 2] numpy
        # rot：Rcw,旋转向量
        
        rot, transl = pose_solver.RANSAC_loop(coords_filtered_2D.astype(np.float64), coords_filtered.astype(np.float64), 256)   # 预测结果,每次取256组点进行PNP
        
        pose_gt = pose.data.numpy()[0,:,:]  # [4 4]
        pose_est = np.eye(4)        # [4 4]
        pose_est[0:3,0:3] = cv2.Rodrigues(rot)[0].T             # Rwc
        pose_est[0:3,3] = -np.dot(pose_est[0:3,0:3], transl)    # twc

        

        transl_err, rot_err = get_pose_err(pose_gt, pose_est)
        rot_err_list.append(rot_err)
        transl_err_list.append(transl_err)
        if transl_err > 0.05 or rot_err>5:
            index = index + 1
        print('step:{}, Pose error: {}m, {}\u00b0，changdu:{}'.format(_ ,transl_err, rot_err,len(coords_filtered_2D))) 


    results = np.array([transl_err_list, rot_err_list]).T   # N 2
    np.savetxt(os.path.join(args.output,
            'pose_err_{}_{}_{}_coord.txt'.format(args.dataset,
            args.scene.replace('/','.'), args.model)), results)
    print('Accuracy: {}%'.format(np.sum((results[:,0] <= 0.050)
                * (results[:,1] <= 5)) * 1. / len(results) * 100))
    print('Median pose error: {}m, {}\u00b0'.format(np.median(results[:,0]), 
            np.median(results[:,1]))) 
    print('Average pose error: {}m, {}\u00b0'.format(np.mean(results[:,0]), 
            np.mean(results[:,1])))
    print('stddev: {}m, {}\u00b0'.format(np.std(results[:,0],ddof=1),
            np.std(results[:,1],ddof=1)))

    accu_res = np.array([[np.median(results[:,0]),np.median(results[:,1])],[np.mean(results[:,0]),np.mean(results[:,1])]])
    np.savetxt('./eval_acc.txt',accu_res)
