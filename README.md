# C3D-tensorflow

This is a repository trying to implement [C3D-caffe][5] on tensorflow, using models directly converted from original C3D-caffe.    
Be aware that there are about 5% video-level accuracy margin on UCF101 split1  between our implement in tensorflow and  the original C3D-caffe. And I modify the code to extract the full/partial features. 
The extract features are used to gan-hash.


## Requirements:

1. Have installed the [tensorflow][1] >= 1.2 version
2. Have installed the [Pytorch][8]
2. You must have installed the following python libs:
a) [Pillow][2]
b) ffmp
3. You must have downloaded the [UCF101][3] (Action Recognition Data Set)
4. Each single avi file is decoded with 25FPS in a single directory.
    - you can use the `./list/convert_video_to_images.sh` script to decode the ucf101 video files
    - run `./list/convert_video_to_images.sh .../UCF101 25` (Note that the fps of UCF101 is 25)
5. Generate {train,test}.list files in `list` directory. Each line corresponds to "image directory" and a class (zero-based). For example:
    - you can use the `./list/convert_images_to_list.sh` script to generate the {train,test}.list for the dataset
    - run `./list/convert_images_to_list.sh .../dataset_images 4`, this will generate `test.list` and `train.list` files by a factor 4 inside the root folder

## Details
1. Assuming you have extracted C3D-tensorflow to your Documents directory on a Linux system such us Ubuntu, your full path will become ~/Documents/C3D-tensorflow. Use the Terminal to get to this directory

2. Now in the C3D-tensorflow folder at ~/Documents/C3D-tensorflow copy and paste the UCF101 folder here so that you get ~/Documents/C3D-tensorflow/UCF101.

3. Navigate to ~/Documents/C3D-tensorflow/list. It has bash files that will not run if you are not logged in as root or using sudo. To overcome this, got to ~/Documents/C3D-tensorflow/list on the terminal and type chmod +x *.sh
.

4. Open the file ~/Documents/C3D-tensorflow/list/convert_video_to_images.sh" with any text editor. You will notice this expression (**if (( $(jot -r 1 1 $2) > 1 )); then** ) on line 31 or so . This piece of code will not run properly unless you have jot installed and this is for Linux only, well as far as I know, so I stand to be corrected. To install jot, in your terminal, type sudo apt-get install athena-jot. You can read more about it here. http://www.unixcl.com/2007/12/jot-print-sequential-or-random-data.html

5. You are now ready to generate imaged from your videos. In the terminal, still in the list directory as ~/Documents/C3D-tensorflow/list, type ./convert_video_to_images.sh ~/Documents/C3D-tensorflow/UCF101 25. You can find the explanation as former mentioned.

You are now ready to generate lists from your images. In the terminal, still in the list directory as ~/Documents/C3D-tensorflow/list, type ./convert_images_to_list.sh ~/Documents/C3D-tensorflow/UCF101 4 You can find the explanation as former mentioned.

Now move out of the list directory into the ~/Documents/C3D-tensorflow and run the train_c3d_ucf101.py file. If you use Python 2 you should not have problems, I guess. However, if you use python 3.5+ you my have to work on the range functions. You will have to convert them to list because in python 3+, ranges are not necessarily converted to list. So you have to do **list(range(......))**

Be sure to also have crop_mean.npy and sports1m_finetuning_ucf101.model files in the ~/Documents/C3D-tensorflow directory if you choose to use the author's codes without modification. 

## Trained models:
|   Model             |   Description     |   Clouds  |  Download   |
| ------------------- | ----------------- |  -------- | ------------|
| C3D sports1M  TF      |C3D sports1M converted from caffe C3D|  Dropbox  |[C3D sports1M ](https://www.dropbox.com/s/zvco2rfufryivqb/conv3d_deepnetA_sport1m_iter_1900000_TF.model?dl=0)       |
| C3D UCF101 TF  |C3D UCF101 trained model converted from caffe C3D|  Dropbox  |[C3D UCF101 ](https://www.dropbox.com/s/u5fxqzks2pkaolx/c3d_ucf101_finetune_whole_iter_20000_TF.model?dl=0 )       |
| C3D UCF101  TF train   |finetuning on UCF101 split1 use C3D sports1M model by  @ [hx173149][7]|  Dropbox  |[C3D UCF101 split1](https://www.dropbox.com/sh/8wcjrcadx4r31ux/AAAkz3dQ706pPO8ZavrztRCca?dl=0)       |
| split1 meanfile  TF   | UCF101 split1 meanfile converted from caffe C3D  |  Dropbox  |[UCF101 split1 meanfile](https://www.dropbox.com/sh/8wcjrcadx4r31ux/AAAkz3dQ706pPO8ZavrztRCca?dl=0)      |
| everything above    |  all four files above  |  baiduyun |[baiduyun](http://pan.baidu.com/s/1nuJe8vn)      |

## Usage

1. Create two folders in C3D ./full_features and ./partial_fetures
2. 'python extractor_full_features.py' and 'extractor_partial_features.py' will extract the full features and partial features respectively.
3. The features will be put in the './full_features' and './partial_features' folders, and you will get two '.txt' files named 'full_feature.list' and 'partial_features.list'.
4. Then, you need to split the feature dataset into train set and test set.(I just can not find the script I wrote before, you can write it or use the defaute splited files for UCF101)
5. run  'CUDA_VISIBLE_DEVICES=&gpu_id python gan_hash_train.py'  to train for partial action retrieval.


## Notes
If you have any problem, no hesitate to let me know (If I have time), I will solve it by my best.


[1]: https://www.tensorflow.org/
[2]: http://pillow.readthedocs.io/en/3.1.x/reference/Image.html
[3]: http://crcv.ucf.edu/data/UCF101.php
[4]: https://github.com/dutran
[5]: https://github.com/facebook/C3D
[6]: http://vlg.cs.dartmouth.edu/c3d/
[7]:https://github.com/hx173149/C3D-tensorflow
[8]:https://github.com/pytorch/pytorch

