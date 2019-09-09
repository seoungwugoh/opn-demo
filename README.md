# Onion-Peel Networks for Deep Video Completion
### Seoung Wug Oh, Sungho Lee, Joon-Young Lee, Seon Joo Kim
### ICCV 2019
[[paper]](https://arxiv.org/abs/1908.08718)

[![Onion-Peel Networks for Deep Video Completion](https://img.youtube.com/vi/bxerAkHAttE/0.jpg)](https://www.youtube.com/watch?v=bxerAkHAttE "Onion-Peel Networks for Deep Video Completion")


### - This repository contains a demo software for OPN with following applications
 1) Reference guided image completion (group photo inpainting)
 2) Video completion


### - Requirements
- python 3.6
- pytorch 0.4.0
- opencv, pillow

### - How to Use
#### Environment setup
```
conda create --name opn python=3.6
source activate opn

pip install opencv-contrib-python pillow

conda install pytorch=0.4.0 cuda90 -c pytorch
conda install torchvision
```

#### Download weights
##### Place it the same folder with demo scripts
```
wget -O OPN.pth "https://www.dropbox.com/s/sxo25p12sfc7na7/OPN.pth?dl=1"
wget -O TCN.pth "https://www.dropbox.com/s/nihciqj551xv7a8/TCN.pth?dl=1"
```

#### Run
##### 1) Group photo inpainting
``` 
python demo_group_image.py --input 3e91f10205_2
```
##### 2) Video inpainting
``` 
python demo_video.py --input parkour
```

#### Test your own images/videos
Prepare your images/videos in ```Image_inputs/[name]``` or ```Video_inputs/[name]```, in the same format and naming rule with the provided examples. 

then, run 
``` 
python demo_group_image.py --input [name]
```
or,
``` 
python demo_video.py --input [name]
```


### - Reference 
If you find our paper and repo useful, please cite our paper. Thanks!
``` 
Onion-Peel Networks for Deep Video Completion
Seoung Wug Oh, Sungho Lee, Joon-Young Lee, Seon Joo Kim
ICCV 2019
```

### - Related Project
Please check out our another approach for video inpaining!
``` 
Copy-and-Paste Networks for Deep Video Inpainting
Sungho Lee, Seoung Wug Oh, DaeYeun Won,  Seon Joo Kim
ICCV 2019
```
[[paper]](https://arxiv.org/abs/1908.11587)
[[github]](https://github.com/shleecs/Copy-and-Paste-Networks-for-Deep-Video-Inpainting)


### - Datasets
This repository partially contains frames sampled from [Youtube-VOS](https://youtube-vos.org/) videos for the group photo application, and [DAVIS](https://davischallenge.org/) videos and masks from [Huang et al.](https://filebox.ece.vt.edu/~jbhuang/project/vidcomp/index.html) for the video inpainting. 


### - Terms of Use
This software is for non-commercial use only.
The source code is released under the Attribution-NonCommercial-ShareAlike (CC BY-NC-SA) Licence
(see [this](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) for details)
