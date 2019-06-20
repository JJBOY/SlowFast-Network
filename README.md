# This is a unofficial implementation of the paper 'SlowFast Networks for Video Recognition ' with Pytorch

## Requirement
 + python 3.6
 + Pytorch 0.1.0
 + opencv
 
## train

### Prepare
First,you need to generate the train list and the test list.The form should like this:
```
related_path_of_video1  frames  label
related_path_of_video2  frames  label
related_path_of_video3  frames  label
...
```
you can use the [genlist.py](./genlist.py) to create them.

This implementation support read the data both in form of video and images.
So maybe you don't need to convert the video into images.You can see the detail in [dataset.py](./dataset.py).

### Run

train
```bash
CUDA_VISIBLE_DEVICES=0 python main.py /train_list_of_your_data /test_list_of_your_data /data_path --gd 20 --lr 0.01 --epoch 60 -b12 -j4
```
the code will use all your visible cuda device. so if you don't want to use all the device, you should use CUDA_VISIBLE_DEVICES 
to tell which device you want to use.

test
```bash
CUDA_VISIBLE_DEVICES=0 python main.py /train_list_of_your_data /test_list_of_your_data /data_path --evaluate --resume your_pretrained_mode_path -b12 -j4
```

### Result

the networks were trained from scratch.
UCF101 only use the split 1.
all the test results are getten from validation split and only use one centre crop.

|      |  top1    |  top5    |
| :----: | :----: | :----: |
|  UCF101    |  55.4%    |  79.0%    |
|  sthsth    |  51.3%    |  79.9%    |

