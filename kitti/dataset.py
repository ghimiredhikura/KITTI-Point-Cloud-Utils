import os
import numpy as np
import torch.utils.data as torch_data

class KittiDataset(torch_data.Dataset):
    def __init__(self, root_dir, split='list', set='sampledata'):
        self.split = split
        self.imageset_dir = os.path.join(root_dir, 'KITTI', 'object', set)
        split_dir = os.path.join(root_dir, 'KITTI', 'object', set, split+'.txt')
        self.image_idx_list = [x.strip() for x in open(split_dir).readlines()]
        self.num_samples = self.image_idx_list.__len__()

        self.lidar_path = os.path.join(self.imageset_dir, 'velodyne')
        self.image_path = os.path.join(self.imageset_dir, 'image_2')
        self.calib_path = os.path.join(self.imageset_dir, 'calib')
        self.label_path = os.path.join(self.imageset_dir, 'label_2')

    def get_image(self, idx):
        image_file = os.path.join(self.image_path, '%.6d.png' % idx)
        assert os.path.exists(image_file)
        return image_file
        #return cv2.imread(image_file) # (H, W, C) -> (H, W, 3) OpenCV reads in BGR mode
    
    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_path, '%.6d.bin' % idx)
        assert os.path.exists(lidar_file)
        return lidar_file
        #return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_path, '%.6d.txt' % idx)
        assert os.path.exists(calib_file)
        return calib_file
        #return Calibration(calib_file)

    def get_labels(self, idx):
        label_file = os.path.join(self.label_path, '%.6d.txt' % idx)
        assert os.path.exists(label_file)
        return label_file       
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        img_file = self.get_image(index)
        lidar_file = self.get_lidar(index)
        label_file = self.get_labels(index)
        calib_file = self.get_calib(index)
        return img_file, lidar_file, label_file, calib_file