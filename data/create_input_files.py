# -----------------------------
# -*- coding:utf-8 -*-
# author:kangkang
# datetime:2019/4/27 12:52
# -----------------------------

from utils.utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco2017',
                       karpathy_json_path='../../../datasets/coco2017/dataset_coco2017.json',
                       image_folder='../../../datasets/coco2017/',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='../../../datasets/coco2017/',
                       max_len=50)
