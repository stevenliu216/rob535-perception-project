# rob535-perception-project
Fall 2019 Final project for ROB535: Self Driving Cars

## Kaggle link
[https://www.kaggle.com/c/rob535-fall-2019-task-1-image-classification](link)

## Task 1
This is the first task for Fall 2019 ROB 535 - Self-driving Cars: Perception and Control. In this task, you are asked to classify a vehicle in a snapshot.

## Task 1 Data
The dataset contains 7360 snapshots, where 5561 of them are used for training and the other 1799 are used for testing. Each snapshot contains an RGB image, a point cloud, the camera matrix, and a list of 3D bounding boxes (only for trainval).

Our task is:

Identify if there is a vehicle in the field of view and within 50m radius of the camera.
If a vehicle is identified within the 50m range, classify it and assign a label among {0,1,2,3} for this snapshot
If there is no vehicle within the 50m range, assign label 0 for this snapshot
All images named xxx_image.jpg has size 1052x1914 with RGB channels. Pointclouds named xxx_cloud.bin have range up to 80m and have calibrated to be mapped to corresponding images by camera matrix named xxx_proj.bin. Each xxx_bbox.bin is a n x 11 matrix with n>=0, and each row represents one 3D bounding box and follows the format:

`[R[0], R[1], R[2], t[0], t[1], t[2], sz[0], sz[1], sz[2], class_id, flag]`

R is a rotation vector (see HW2 Q3.2). t is a translation vector of bounding box's centroid. sz are (length, width, height) of bounding box. class_id can be mapped to label of snapshot by classes.csv. flag indicates whether this bounding box is ignored for training. If the centroid of this bounding box is outside of 50m radius, or it is occluded, the bounding box will be ignored for classification and the flag is set to 1.

Both Python3 and Matlab scripts are provided on Canvas to decode and visualize those data.
