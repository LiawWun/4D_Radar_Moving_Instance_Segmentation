# 4D_Radar_Moving_Instance_Segmentation


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

ITRI 4D radar dataset can be download from this link(to be add).

# Code reference
The pytorch implementation of [flownet3d](https://github.com/xingyul/flownet3d) based on [WangYueFt/dcp](https://github.com/WangYueFt/dcp), [sshaoshuai/Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch) and [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
