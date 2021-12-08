from __future__ import division

import torch
import numpy as np
import random
from imgaug import augmenters as iaa

# depth：深度图[480 640]
# 返回深度图对齐到RGB图后，RGB相应的深度信息
def get_depth(depth, calibration_extrinsics, intrinsics_color,
              intrinsics_depth_inv):
    """Return the calibrated depth image (7-Scenes). 
    Calibration parameters from DSAC (https://github.com/cvlab-dresden/DSAC) 
    are used.
    """
    '''
    利用深度摄像头内参矩阵把深度平面坐标（深度图坐标）转换到深度摄像头空间坐标，
    再利用外参计算旋转矩阵和平移矩阵，把深度摄像头空间坐标转换到RGB摄像头空间坐标，
    最后利用RGB摄像头内参矩阵把RGB摄像头空间坐标转换到RGB平面坐标（RGB图坐标）。
    这里只记录一下最终测试程序的思路：
    '''
    img_height, img_width = depth.shape[0], depth.shape[1]
    depth_ = np.zeros_like(depth)       # [480 640]
    x = np.linspace(0, img_width-1, img_width)      # 640
    y = np.linspace(0, img_height-1, img_height)    # 480
    
    xx, yy = np.meshgrid(x, y)          # 坐标网格化[img_width img_height]
    xx = np.reshape(xx, (1, -1))        # [1, img_width*img_height]
    yy = np.reshape(yy, (1, -1))        # [1, img_width*img_height]
    ones = np.ones_like(xx)             # [1, img_width*img_height]
    
    pcoord_depth = np.concatenate((xx, yy, ones), axis=0)   # [3, img_width*img_height], 像素坐标
    depth = np.reshape(depth, (1, img_height*img_width))    # [1, img_width*img_height]
    
    ccoord_depth = np.dot(intrinsics_depth_inv, pcoord_depth) * depth   # 像素坐标-->归一化坐标-->相机坐标[3, img_width*img_height]

    ccoord_depth[1,:] = - ccoord_depth[1,:]
    ccoord_depth[2,:] = - ccoord_depth[2,:]
    
    ccoord_depth = np.concatenate((ccoord_depth, ones), axis=0) # [4, img_width*img_height]
    ccoord_color = np.dot(calibration_extrinsics, ccoord_depth) # [3, img_width*img_height],RGB相机坐标
   
    ccoord_color = ccoord_color[0:3,:]
    ccoord_color[1,:] = - ccoord_color[1,:]
    ccoord_color[2,:] = depth

    pcoord_color = np.dot(intrinsics_color, ccoord_color)       # RGB像素坐标*Z
    pcoord_color = pcoord_color[:,pcoord_color[2,:]!=0]
    
    pcoord_color[0,:] = pcoord_color[0,:]/pcoord_color[2,:]+0.5 # RGB像素坐标
    pcoord_color[0,:] = pcoord_color[0,:].astype(int)
    pcoord_color[1,:] = pcoord_color[1,:]/pcoord_color[2,:]+0.5
    pcoord_color[1,:] = pcoord_color[1,:].astype(int)
    pcoord_color = pcoord_color[:,pcoord_color[0,:]>=0]
    pcoord_color = pcoord_color[:,pcoord_color[1,:]>=0]
    
    pcoord_color = pcoord_color[:,pcoord_color[0,:]<img_width]
    pcoord_color = pcoord_color[:,pcoord_color[1,:]<img_height]

    depth_[pcoord_color[1,:].astype(int),
           pcoord_color[0,:].astype(int)] = pcoord_color[2,:]
    return depth_

def get_coord(depth, pose, intrinsics_color_inv, dataset):
    """Generate the ground truth scene coordinates from depth and pose.
    """
    img_height, img_width = depth.shape[0], depth.shape[1]  # 480 640
    mask = np.ones_like(depth)
    mask[depth==0] = 0                                          # 深度为0处，数值为0，否则为1
    mask = np.reshape(mask, (img_height, img_width,1))          # [480 640 1]
    x = np.linspace(0, img_width-1, img_width)
    y = np.linspace(0, img_height-1, img_height)
    
    xx, yy = np.meshgrid(x, y)
   
    xx = np.reshape(xx, (1, -1))        # [1, 640*480]
    yy = np.reshape(yy, (1, -1))        # [1, 640*480]
    ones = np.ones_like(xx)             # [1, 640*480]
    pcoord = np.concatenate((xx, yy, ones), axis=0)     # [3, 640*480],像素坐标
    
    depth = np.reshape(depth, (1, img_height*img_width))        # [1, 640*480]
    ccoord = np.dot(intrinsics_color_inv, pcoord) * depth       # 相机坐标 [3 640*480]
    ccoord = np.concatenate((ccoord, ones), axis=0)             # 相机坐标 [4 640*480]
    
    # if dataset == 'my':
    #     scoord  = np.dot(np.swapaxes(ccoord,0,1), pose)
    # else:
    scoord = np.dot(pose, ccoord)               # 世界坐标 [3 640*480]
    scoord = np.swapaxes(scoord,0,1)            # 世界坐标 [640*480 3]


    scoord = scoord[:,0:3]
    scoord = np.reshape(scoord, (img_height, img_width,3))      # 世界坐标 [480 640 3]
    scoord = scoord * mask
    mask = np.reshape(mask, (img_height, img_width))            # [480 640]
    
    return scoord, mask

# 数据增强操作
def data_aug(img, coord, mask, aug=True):
    img_h, img_w = img.shape[0:2]
    if aug:
        trans_x = random.uniform(-0.2,0.2)              # 平移
        trans_y = random.uniform(-0.2,0.2)

        aug_add = iaa.Add(random.randint(-20,20))

        scale=random.uniform(0.7,1.5)                   # 缩放
        rotate=random.uniform(-30,30)                   # 旋转
        shear=random.uniform(-10,10)                    # 裁剪

        aug_affine = iaa.Affine(scale=scale,rotate=rotate,
                    shear=shear,translate_percent={"x": trans_x, "y": trans_y}) 
        aug_affine_lbl = iaa.Affine(scale=scale,rotate=rotate,
                    shear=shear,translate_percent={"x": trans_x, "y": trans_y},
                    order=0,cval=1) 
        img = aug_add.augment_image(img) 
    else:
        trans_x = random.randint(-3,4)
        trans_y = random.randint(-3,4)
    
        aug_affine = iaa.Affine(translate_px={"x": trans_x, "y": trans_y}) 
        aug_affine_lbl = iaa.Affine(translate_px={"x": trans_x, "y": trans_y},
                    order=0,cval=1) 
                
    padding = torch.randint(0,255,size=(img_h,
                            img_w,3)).data.numpy().astype(np.uint8)
    padding_mask = np.ones((img_h,img_w)).astype(np.uint8)  
       
    img = aug_affine.augment_image(img)
    
    coord = aug_affine.augment_image(coord)
    mask = aug_affine.augment_image(mask)
    mask = np.round(mask)
    #lbl = aug_affine_lbl.augment_image(lbl)
    padding_mask = aug_affine.augment_image(padding_mask)
    img = img + (1-np.expand_dims(padding_mask,axis=2)) * padding

    return img, coord, mask



# img [480 640 3]
# coord_img [60, 80, 3]
# mask [60 80]
def to_tensor(img, coord_img, mask):
    img = img.transpose(2, 0, 1)                        
    coord_img = coord_img.transpose(2, 0, 1)   
    img = img / 255.        
    img = img * 2. - 1.
    coord_img = coord_img / 1000.  
    img = torch.from_numpy(img).float()
    coord_img = torch.from_numpy(coord_img).float()
    mask = torch.from_numpy(mask).float()
    return img, coord_img, mask


def to_tensor_query(img, pose):
    img = img.transpose(2, 0, 1)    
    img = img / 255.
    img = img * 2. - 1.
    img = torch.from_numpy(img).float()
    pose = torch.from_numpy(pose).float()
    return img, pose
