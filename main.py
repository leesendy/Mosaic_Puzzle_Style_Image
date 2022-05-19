# -*- coding: UTF-8 -*-
'''
-----------------------------------
  File Name:   mosaic_puzzle
  Description: Build mosaic_puzzle style image
  Author:      lidisen
  Date:        2022/5/17
-----------------------------------
  Change Activity: 2022/5/17
'''
import argparse
import os, cv2, math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

def GetRGBMean(img):
    '''Calculate RGB channel means'''
    mr = img[:, :, 0].mean()
    mg = img[:, :, 1].mean()
    mb = img[:, :, 2].mean()
    return mr, mg, mb

def GetMeanSUM(img):
    '''Calculate the sum of the RGB channel means'''
    mr = img[:, :, 0].mean()
    mg = img[:, :, 1].mean()
    mb = img[:, :, 2].mean()
    return mr + mg + mb

def GetMeanT_PathDict(matched_img_rootpath, groups):
    '''
    Divide all the matched images into groups based means
    :return meanT_path_dict--dictionary
    { min_mean : (img_mean, img_path),
    min_mean+stride: (img2_mean, img2_path),
    ...
    max_mean:(imgN_mean, imgN_path)}
    '''
    means_path_list = list()
    for name in os.listdir(matched_img_rootpath):
        sub_img_path = os.path.join(matched_img_rootpath, name)
        img = cv2.imread(sub_img_path)
        mean = GetMeanSUM(img)
        means_path_list.append((mean, sub_img_path))
    sorted_means_path = sorted(means_path_list, key=lambda x:x[0])
    min_mean = sorted_means_path[0][0]
    max_mean = sorted_means_path[-1][0]
    stride_mean = math.floor((max_mean - min_mean) / groups)
    meanT_path_dict = dict()
    for i in range(int(min_mean), int(max_mean), stride_mean):
        temp = []
        for j in sorted_means_path:
            if i <= j[0] <= (i+stride_mean):
                temp.append(j)
        if len(temp) > 0:
            meanT_path_dict[i] = temp
    return meanT_path_dict

def CompareSim(img1, img2):
    '''Calculate similarity between two images'''
    mr, mg, mb = GetRGBMean(img1)
    m1r, m1g, m1b = GetRGBMean(img2)
    return np.abs(mr - m1r) + np.abs(mg - m1g) + np.abs(mb - m1b)

def GetPathsByMean(steam, mean):
    '''Rough match, get available images' paths'''
    sub_img_list = []
    for i, item in enumerate(steam.keys()):
        if mean >= item and i < len(steam.keys()) - 1 and mean < [*steam.keys()][i + 1]:
            sub_img_list = steam[item]
        if i == len(steam.keys()) - 1 and mean >= item:
            sub_img_list = steam[item]
        if mean < item and i == 0:
            sub_img_list = steam[item]
        if mean > item and i == len(steam.keys()) - 1:
            sub_img_list = steam[item]
    ava_path_list = []
    for i in sub_img_list:
        ava_path_list.append(i[1])
    return ava_path_list

def GetSimImg_Path(img, means_dict, param_px):
    '''Get the final matched image and path'''
    mean = GetMeanSUM(img)
    match_path_list = GetPathsByMean(means_dict, mean)
    degree_list = []
    path_list = []
    for i, match_path in enumerate(match_path_list):
        match_img = cv2.imread(match_path)
        degree = CompareSim(img, match_img)
        degree_list.append(degree)
        path_list.append(match_path)

    sorted_degree_list = sorted(degree_list)
    value_list = sorted_degree_list[0:20]
    random.shuffle(value_list)
    index = degree_list.index(value_list[0])
    select_data_path = path_list[index]

    select_img = cv2.imread(select_data_path)
    h = select_img.shape[0]
    w = select_img.shape[1]
    if h > w:
        select_img = select_img[:w, :, :]
    else:
        select_img = select_img[:, :h, :]

    select_img = cv2.resize(select_img, dsize=(param_px, param_px), interpolation=cv2.INTER_NEAREST)
    return (select_img, select_data_path)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_img_path', type=str, default='./target_data/target_0.jpeg', help='Target images path')
    parser.add_argument('--file_format', type=str, default='.jpeg', help='Target image file format')
    parser.add_argument('--match_img_path', type=str, default='./match_data_cifar10', help='Match images path')
    parser.add_argument('--groups', type=int, default=10, help='Divide match images into N groups')
    parser.add_argument('--gain', type=int, default=3, help='Enlarge the original image by N times')
    parser.add_argument('--param_px', type=int, default=30, help='Mosaic block size')
    parser.add_argument('--save_excel', type=bool, default=False, help='Save match img locations in final image')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    # 基于图像RGB均值，将所有图像分成若干组
    # Group matching images based on means
    mean_path_dict = GetMeanT_PathDict(opt.match_img_path, opt.groups)

    # 原图放大
    # Enlarge original image
    file_format = opt.file_format
    target_img_name = os.path.basename(opt.target_img_path).split(file_format)[0]
    org_img = cv2.imread(opt.target_img_path)
    org_img = cv2.resize(org_img, dsize=(org_img.shape[1] * opt.gain, org_img.shape[0] * opt.gain),
                         interpolation=cv2.INTER_NEAREST)
    param_px = opt.param_px
    iterNum_row = math.floor(org_img.shape[0] / param_px)
    iterNum_col = math.floor(org_img.shape[1] / param_px)

    # 遍历图像，获取对应子块的匹配图像和图像地址
    # Get the matching image and path of the corresponding patch_block
    select_img_list, select_path_list, select_local_list = [], [], []
    start_h = 0
    for r in tqdm(range(iterNum_row)):
        start_w = 0
        for c in range(iterNum_col):
            patch_img = org_img[start_h:start_h + param_px, start_w:start_w + param_px, :]
            select_img, select_path = GetSimImg_Path(patch_img, mean_path_dict, param_px)
            select_img_list.append(select_img)
            select_path_list.append(select_path)
            select_local_list.append((r, c))
            start_w += param_px
        start_h += param_px

    # 匹配图像合成大图
    # The selected matching images form a larger image
    start_idx = 0
    row_img_list = []
    for _ in range(iterNum_row):
        row_img_list.append(np.hstack(select_img_list[start_idx:start_idx + iterNum_col]))
        start_idx += iterNum_col
    out_img_np = np.vstack(row_img_list)
    print('Output img shape', out_img_np.shape)

    # 保存马赛克拼图以及组成图像地址和所在图像位置
    # Save the mosaic_puzzle image and selected matching images' path and location
    timestamp = str(time.strftime('%Y-%m-%d-%H-%M', time.localtime()))
    output_img_name = target_img_name + '_mosaic_' + timestamp + '.jpg'
    target_rootpath = os.path.dirname(opt.target_img_path)
    output_path = os.path.join(target_rootpath, output_img_name)
    cv2.imwrite(output_path, out_img_np)

    if opt.save_excel:
        output_location_name = target_img_name + '_location_' + timestamp+ '.xlsx'
        df = pd.DataFrame({'location': select_local_list, 'path': select_path_list})
        df.to_excel(os.path.join(target_rootpath,output_location_name))
    print('Congratulations, Mosaic_Puzzle Images Had Saved in ', output_path)
