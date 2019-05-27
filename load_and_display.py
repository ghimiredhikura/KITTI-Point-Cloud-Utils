import numpy as np
import os
import cv2
import kitti.kitti_util as utils
import kitti.mayavi_viewer as mview
import torch.utils.data as torch_data
from kitti.dataset import KittiDataset
import torch
import kitti.kitti_aug_utils as aug_utils
import kitti.bev_utils as bev_utils
import kitti.config as cnf

root_dir = 'data/'
dataset = KittiDataset(root_dir=root_dir, split='list', set='sampledata')
data_loader = torch_data.DataLoader(dataset, batch_size=1, shuffle=False)

for batch_index, (image_file, lidar_file, label_file, calib_file) in enumerate(data_loader):

    for i, img_file in enumerate(image_file):
        img = cv2.imread(image_file[i])
        objects = utils.read_label(label_file[i])
        lidar = np.fromfile(lidar_file[i], dtype=np.float32).reshape(-1, 4)
        calib = utils.Calibration(calib_file[i])

        lidar = bev_utils.removePoints(lidar, cnf.boundary)        
        img_bev = bev_utils.makeBVFeature(lidar, cnf.DISCRETIZATION, cnf.boundary)
        img_bev = np.uint8(img_bev*255)
        labels, noObjectLabels = bev_utils.read_labels_for_bevbox(objects)
        print(labels)
        if not noObjectLabels:
            labels[:, 1:] = aug_utils.camera_to_lidar_box(labels[:, 1:])  # convert rect cam to velo cord
        target = bev_utils.build_yolo_target(labels, cnf.boundary)
        
        invlabels = bev_utils.inverse_yolo_target(target, cnf.boundary)
        if invlabels.shape[0]:
            invlabels[:, 1:] = aug_utils.lidar_to_camera_box(invlabels[:, 1:]) # convert velo to rect cam cord
        print(invlabels)


        bev_utils.draw_box_in_bev(img_bev, target)
        cv2.imshow("img_bev", img_bev)

        pc = lidar[:, 0:3]
        mview.show_lidar_with_boxes(pc, objects, calib, img_fov=False, img_width=img.shape[1], img_height=img.shape[0], fig=None)

        img = mview.show_image_with_boxes(img, objects, calib, False)
        cv2.imshow("img", img)

        if cv2.waitKey(0) & 0xff == 27:
            break