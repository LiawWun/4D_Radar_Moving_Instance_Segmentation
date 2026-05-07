# 4D_Radar_Moving_Instance_Segmentation

Motion Segmentation (Seq 15)
![Motion Segmentation](/demo_video/Motion_Segmentation_Demo.gif)

Moving Instance Segmentation (Seq 11)
![Moving Instance Segmentation](/demo_video/Radar_Moving_Instance_Segmentation_Demo.gif)




# Environment setting

```bash
conda create -n flownet -y
conda activate flownet
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install numpy==1.24.1
pip install scipy
pip install h5py
pip install tqdm

cd lib
python setup.py install
cd ../
```
# Dataset download

The ITRI 4D Radar Dataset can be downloaded from the following links:

- **Radar point cloud only**: TO BE ADD.  
- **Full dataset** (including raw bag data, camera data, LiDAR data, pose information, and LiDAR bounding box annotations):  TO BE ADD.

After downloading the dataset, organize the folder structure as follows:

```bash
dataset_folder/
├── seq0/
├── seq1/
├── seq2/
├── seq3/
├── ...
└── seq23/
```

Then, modify the `self.base_folder` path in `/dataset/itri_dataset.py` to point to your dataset directory:

```python
class ITRI_Dataset(Dataset):

    def __init__(self, mode='train', transform=None, num_frame=2):

        self.mode = mode
        self.num_frame = num_frame
        assert 2 <= self.num_frame <= 5, "num_frame must be between 2 and 5 (inclusive)."
        self.transform = transform

        self.base_folder = "/your/path/here/"
        self.train_seq = ["seq0", "seq3", "seq4", "seq5", "seq6", "seq7", "seq9", "seq12", "seq13", "seq14", "seq16", "seq17", "seq18", "seq19", "seq20", "seq21", "seq22", "seq23"]
        self.val_seq   = ["seq1", "seq2", "seq8", "seq11", "seq15"]

```


# Radar Point Cloud Fields

The input radar point cloud is cropped to a sensing range of 50 meters. 
Each point consists of the following 12 fields:

1. **x, y, z**: 3D position coordinates  
2. **RCS**: Radar Cross Section (RCS) measurement  
3. **v_r**: Raw Doppler velocity measurement  
4. **v_r_compensated**: Ego-motion compensated Doppler velocity. The ego velocity used for Doppler compensation is calculated based on [this paper](https://ieeexplore.ieee.org/abstract/document/6728341).  
5. **moving_mask**: Binary mask indicating whether the point is moving (`1`) or static (`0`)  
6. **cluster_id**: Cluster ID for moving objects. Static points are assigned `-1`  
7. **xc, yc, zc**: Center position of the associated moving cluster. Static points are assigned `(0, 0, 0)`  
8. **time_idx**: Temporal index of the frame. `0` represents the current frame, `1` represents frame `t-1`, and so on.

# Training
To train the model, run the following command in the terminal:
```bash
python train.py
```
After training starts, a results folder will be automatically created. The folder name indicates the time at which the training process was started. The folder also stores the top 5 model weights based on their performance on the validation dataset.

# Evaluation
To evaluation the model, run the following command in the terminal:
```bash
python inference.py xxxx-xx-xx_xx-xx
```
Here, xxxx-xx-xx_xx-xx represents the timestamp of when the training process was started. The script will evaluate the saved top 5 model weights and report the best score.

# Reproducibility

Some operations in the code are inherently non-deterministic (e.g., [`atomicAdd`](https://glaringlee.github.io/notes/randomness.html#:~:text=There%20are%20some%20PyTorch%20functions,order%20of%20additions%20being%20nondetermnistic.)) and cannot be fully controlled by random seeds. Therefore, slight performance variations may be observed. Results may also vary across different GPUs and CUDA versions.

# Future work
A potential direction for this work is transitioning to a **self-supervised learning** framework to overcome the **LiDAR labeling bottleneck**. Since LiDAR becomes too sparse at long ranges for manual annotation, we aim to use [this paper](https://ieeexplore.ieee.org/abstract/document/6728341) to generate pseudo-label to train on unlabeled far-field data. This would allow the method to fully leverage the long-range sensing capabilities of 4D Radar beyond the limits of LiDAR-based ground truth.

# Code reference
The pytorch implementation of [flownet3d](https://github.com/xingyul/flownet3d) based on [WangYueFt/dcp](https://github.com/WangYueFt/dcp), [sshaoshuai/Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch) and [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
