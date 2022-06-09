#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:17:20 2019

@author: xingyu
"""

from imutils import paths
import numpy as np
import cv2
import os
import argparse
import random
import json

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


parser = argparse.ArgumentParser(description='crop the licence plate from original image')
parser.add_argument("-image", help='image path', default='/data/zjw/data/ccpd_green/images/json_test/val/', type=str)
parser.add_argument("-dir_train", help='save directory', default='train', type=str)
parser.add_argument("-dir_val", help='save directory', default='/data/zjw/data/json_true/val/', type=str)
args = parser.parse_args()

img_paths = []
img_paths += [el for el in paths.list_images(args.image)]
random.shuffle(img_paths)

save_dir_train = args.dir_train
save_dir_val = args.dir_val

print('image data processing is kicked off...')
print("%d images in total" % len(img_paths))

idx = 0
idx_train = 0
idx_val = 0
for i in range(len(img_paths)):
    filename = img_paths[i]
    filename = os.path.basename(filename)
    file, name = os.path.splitext(filename)
    print(file)
    filejson = '/data/zjw/data/ccpd_green/images/json_test/label_test/' + file + '.json'
    print(filejson)
    with open(filejson, 'r', encoding='utf-8')as fp:
        json_data = json.load(fp)
        shape = json_data['shapes']
        points = shape[0]['points']
        x1, y1, x2, y2 = points[0][0], points[0][1], points[1][0], points[1][1]
        print(x1, y1, x2, y2)
        w = int(x2 - x1 + 1.0)
        h = int(y2 - y1 + 1.0)

        img = cv2.imread('/data/zjw/data/ccpd_green/images/json_test/val/' + file + '.jpg')
        print('/data/zjw/data/ccpd_green/images/json/train/' + file + '.JPG')
        img_crop = np.zeros((h, w, 3))
        img_crop = img[int(y1):int(y2+1), int(x1):int(x2+1), :]
    #    img_crop = cv2.resize(img_crop, (94, 24), interpolation=cv2.INTER_LINEAR)

        # pre_label = imgname_split[4].split('_')XA
        # lb = ""
        # lb += provinces[int(pre_label[0])]
        # lb += alphabets[int(pre_label[1])]
        # for label in pre_label[2:]:
        #     lb += ads[int(label)]
        # print(lb)
        lb = shape[0]['label']

        idx += 1

        if idx % 100 == 0:
            print("%d images done" % idx)

        # if idx % 4 == 0:
        save = save_dir_val + '/' + lb + '_' + filename + '.jpg'
        cv2.imwrite(save, img_crop)
        idx_val += 1
        # else:
        #     save = save_dir_train+'/'+lb+suffix
        #     cv2.imwrite(save, img_crop)
        #     idx_train += 1
        
print('image data processing done, write %d training images, %d val images' % (idx_train, idx_val))
