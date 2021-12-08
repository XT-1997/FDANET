'''
    绘制chess/seq-03相机轨迹
'''
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

sys.path.append("../../dataset/chess")
if os.path.exists("../../dataset/chess/seq-03"):
    print("OK")

gt_pose = [];
file_list = os.listdir("../../dataset/chess/seq-03")
file_list.sort(key=lambda x : x[6:12])      # os.listdir读取的文件顺序是无序的，先进行排序
print(file_list)

for file in file_list:
    if file[-3:] == 'txt':
        context = np.loadtxt(os.path.join("../../dataset/chess/seq-03",file))
        true_pose = context[:3,3]
        gt_pose.append(true_pose)
pose = np.vstack(gt_pose)

est_pose = np.loadtxt("../data/7Scenes/perdict_pose.txt")


ax = plt.figure().add_subplot(111, projection='3d')
x,y,z = pose[:,0],pose[:,1],pose[:,2]
ax.plot3D(x,y,z,color="blue",linewidth=1)
# ax.scatter(x,y,z,c='b',s=0.05)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

est_x,est_y,est_z = est_pose[:,0],est_pose[:,1],est_pose[:,2]
# ax.scatter(est_x,est_y,est_z,c='g',s=5)
ax.plot3D(est_x,est_y,est_z,color="green",linewidth=1)
plt.show()

