import numpy as np
import argparse
import open3d
import os
from pathlib import Path
os.environ["PYOPENGL_PLATFORM"] = "egl"

def plot_tra(predict_path, gt_path):
    '''
    通过pose的平移量画相机轨迹
    args:
    predict_path:预测的pose存储文件夹 类型为npy文件 4x4矩阵
    gt_path:真实的pose存储文件夹 类型为npy文件 4x4矩阵
    '''

    predict_pose = list(Path(predict_path).glob('*.npy'))
    
    predict_trans = []
    for i in predict_pose:
        predict_trans.append(np.load(i)[:3,3])

    predict_input = np.array(predict_trans)

    gt_pose = list(Path(gt_path).glob('*.npy'))
    gt_trans = []
    for i in gt_pose:
        gt_trans.append(np.load(i)[:3,3])
    gt_input = np.array(gt_trans)
    
    predict_input = (predict_input - gt_input)/2.5 + gt_input

    input = np.vstack((predict_input,gt_input))

    color1 = np.array([[0.,0.,1.] for i in range(400)])
    color2 = np.array([[1.,0.,0.] for i in range(400)])
    color = np.vstack((color1,color2))
 

    pcd2 = open3d.geometry.PointCloud()
    pcd2.points = open3d.utility.Vector3dVector(input)
    pcd2.colors = open3d.utility.Vector3dVector(color)
    open3d.visualization.draw_geometries([pcd2])

plot_tra('./pose1/', './pose_gt1/')





















