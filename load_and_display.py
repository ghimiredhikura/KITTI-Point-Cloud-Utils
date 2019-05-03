import numpy as np
import cv2
import kitti.kitti_util as utils
import kitti.mayavi_viewer as mview
import torch.utils.data as torch_data
from kitti.dataset import KittiDataset
import torch

root_dir = 'F:/charm-torch/projects/kitti_utils/data'
dataset = KittiDataset(root_dir=root_dir, split='list', set='sampledata')
data_loader = torch_data.DataLoader(dataset, batch_size=5, shuffle=False)

for batch_index, (image_file, lidar_file, label_file, calib_file) in enumerate(data_loader):

    for i, img_file in enumerate(image_file):
        img = cv2.imread(image_file[i])
        label = utils.get_label(label_file[i])
        lidar = np.fromfile(lidar_file[i], dtype=np.float32).reshape(-1, 4)
        calib = utils.Calibration(calib_file[i])
        
        pc = lidar[:, 0:3]
        objects = utils.read_label(label_file[i])
        print(img.shape[0], img.shape[1])
        mview.show_lidar_with_boxes(pc, objects, calib, img_fov=True, img_width=img.shape[1], img_height=img.shape[0], fig=None)
        cv2.imshow("img", img)
        cv2.waitKey(0)