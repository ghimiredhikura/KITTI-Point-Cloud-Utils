import numpy as np
import os
import math
import cv2
import kitti.kitti_util as utils
import kitti.mayavi_viewer as mview
import torch.utils.data as torch_data
from kitti.dataset import KittiDataset
import torch
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

        if not noObjectLabels:
            labels[:, 1:] = utils.camera_to_lidar_box(labels[:, 1:])  # convert rect cam to velo cord
        target = bev_utils.build_yolo_target(labels, cnf.boundary)

        pc = lidar[:, 0:3]
        # display point cloud, point cloud bev, and corresponting front camera 2d image all with object bbox. 
        # note that in 2d image object bbox is projected from point cloud 3d labels.
        mview.show_lidar_with_boxes(pc, objects, calib, img_fov=False, img_width=img.shape[1], img_height=img.shape[0], fig=None)
        img = mview.show_image_with_boxes(img, objects, calib, False)
        bev_utils.draw_box_in_bev(img_bev, target)
        cv2.imshow("LIDAR PC BEV", img_bev)
        cv2.imshow("FRONT CAM 2D IMAGE", img)
        
        # Convert bev object labels back to lidar 3d box. 
        invlabels = bev_utils.inverse_yolo_target(target, cnf.boundary)
        if invlabels.shape[0]:
            invlabels[:, 1:] = utils.lidar_to_camera_box(invlabels[:, 1:]) # convert velo to rect cam cord

        # objects_new is the bbox in lidar point cloud projected from bev bbox. 
        # You can use peace of code if you are working with bev and want to display result back to lidar cloud or 2d image. 
        objects_new = []
        for i, l in enumerate(invlabels):

            str = "Pedestrian"
            if l[0] == 0:str="Car"
            elif l[0] == 1:str="Pedestrian"
            elif l[0] == 2: str="Cyclist"
            else:str = "DontCare"
            line = '%s 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0' % str

            obj = utils.Object3d(line)
            obj.t = l[1:4]
            obj.h,obj.w,obj.l = l[4:7]
            obj.ry = np.arctan2(math.sin(l[7]), math.cos(l[7]))
            objects_new.append(obj)

        #for obj in objects_new: 
        #    print( "+ %d - " % batch_index, obj.to_kitti_format())

        #mview.show_lidar_with_boxes(pc, objects_new, calib, img_fov=False, img_width=img.shape[1], img_height=img.shape[0], fig=None)
        #img = mview.show_image_with_boxes(img, objects_new, calib, False)
        
        if cv2.waitKey(0) & 0xff == 27:
            break