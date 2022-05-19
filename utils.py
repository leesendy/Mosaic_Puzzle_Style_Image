# -*- coding: UTF-8 -*-
'''
-----------------------------------
  File Name:   utils
  Description: Convert cifar10 dataset to .jpg data
  Author:      lidisen
  Date:        2022/5/19
-----------------------------------
  Change Activity: 2022/5/19
'''

import sys, os
from six.moves import cPickle
import matplotlib.pyplot as plt
from tqdm import tqdm
def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.

    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.

    # Returns
        A tuple `(data, labels)`.
    """
    with open(fpath, 'rb') as f:
        if sys.version_info < (3,):
            d = cPickle.load(f)
        else:
            d = cPickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels

path = './data_sourse/test_batch'
save_path = './match_data_cifar10'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

x, _ = load_batch(path)
x_trans = x.transpose(0, 2, 3, 1)

for idx in tqdm(range(10000)):
    file_name = 'cifar10-' + str(idx) + '.jpg'
    plt.imsave(os.path.join(save_path, file_name), x_trans[idx])
print('------end-------')