import numpy as np 
import cv2
def get_pose_err(pose_gt, pose_est):
    transl_err = np.linalg.norm(pose_gt[0:3,3]-pose_est[0:3,3])
    rot_err = pose_est[0:3,0:3].T.dot(pose_gt[0:3,0:3])
    rot_err = cv2.Rodrigues(rot_err)[0]         # 旋转向量 [3 1]
    rot_err = np.reshape(rot_err, (1,3))        # 旋转向量 [1 3]
    rot_err = np.reshape(np.linalg.norm(rot_err, axis = 1), -1) / np.pi * 180. # 二范数即转角
    return transl_err, rot_err[0]
