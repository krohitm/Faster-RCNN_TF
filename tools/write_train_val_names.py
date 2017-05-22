#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 20:16:46 2017

@author: krohitm
"""

import numpy as np

num_train_ims = 222222

cur_filename = np.zeros(num_ims)
person_cur_filename = np.zeros((num_ims, 2))
orig_indices = np.zeros(num_ims)

for i in range(num_ims):
    person_cur_filename[i, 1] = '1'
    orig_indices[i] = i
    cur_filename[i,0] = (str(i+1)).zfill(7)
    person_cur_filename[i,0] = (str(i+1)).zfill(7)

shuffle_indices = np.random.permutation(num_ims)
train_indices = 