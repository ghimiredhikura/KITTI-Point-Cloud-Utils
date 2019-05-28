## KittiPointCloudUtils

This repo serves as kitti point cloud dataset utilities. Following are the main functionality.
I am sharing this because if someone is working with kitti point cloud for the first time, this one is going to save your a lot of time. 

Thank me later. ^^ 

1. *Load Point Cloud and filter out points that are not in range of front camera fov*  
2. *Display Point Cloud with object bbox using mayavi library*  
3. *BEV Conversion and display with object bbox using opencv library*  
4. *Project 3D bbox from lidar to camera and camera to lidar box*  

#### Original Point Cloud with object bbox

![Alt text](data/assets/point_cloud.png)

#### Point Cloud BEV with object bbox

![Alt text](data/assets/bird_eye_view.png)

#### Front Left Camera image with object bbox projected from Point Cloud labels

![Alt text](data/assets/front_cam_image.png)
