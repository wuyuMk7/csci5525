# CSCI 5525 Final Project

# 0.06 for v1 v2 200/800 epochs batch 512
# 0.002 for v3 300 epochs batch 512
batch 1024 with 0.002
(thinking to investigate batch 2048 with 0.002, 0.0001)

## How to run this project

1. Clone the project to your local computer and load the submodules
'''
git clone <link to this repo>
cd <folder for this repo>
git submodule init
git submodule update
'''

2. Install dependencies. Dependencies can be found in the requirements file.

3. 

## Results

### Model Accuracy:
- v1-200: 82.1
- v1-800: 84.28
- v2-200: 80.26
- v2-800: 88.62
- v3-512-0.002: 64.33
- v3-1024-0.002: 66.19
- v3-2048-0.002: 64.88
- v3-2048-0.0001: 48.37
- v3-2048-fix-0.002: 71.07
- v3-2048-fix-0.0001: 51.85

Pre-trained models can be found in the following link:
https://drive.google.com/drive/folders/1CC_NHFIzYTx5E6IeMVQB955yXOqatMML?usp=sharing
*Please switch to your UMN account if you don't have permission to access it.

## References

Papers for MoCo-v1, MoCo-v2, and MoCo-v3.

https://github.com/facebookresearch/moco
https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
https://github.com/CupidJay/MoCov3-pytorch
https://github.com/searobbersduck/MoCo_v3_pytorch

