#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 14:48:19 2017

@author: krohitm
"""
import numpy as np

from xml.etree import ElementTree as ET


if __name__=='__main__':
    tree = ET.parse('000007.xml')
    root = tree.getroot()
    
    bbox = np.array(np.loadtxt(
            '/data0/krohitm/object_boundaries/person_median.csv', dtype=str, 
            delimiter=',', skiprows=1, ))

    data_dict = {}
    data_dict['flickrid'] = 'krm'
    data_dict['name'] = 'krohitm'
    data_dict['pose'] = 'unknown'
    data_dict['width'] = '244'
    data_dict['height'] = '244'
    data_dict['depth'] = '3'
    num_imgs = len(bbox[:,0])
    for i in range(num_imgs):
        data_dict['filename'] = '{0}.jpg'.format(str(i+1).zfill(7))
        data_dict['xmin'] = bbox[i,1]
        data_dict['ymin'] = bbox[i,2]
        data_dict['xmax'] = bbox[i,3]
        data_dict['ymax'] = bbox[i,4]
        for node in root.iter():
            if node.tag in data_dict:
                node.text = data_dict[node.tag]
        #tree.write('/home/krohitm/code/Faster-RCNN_TF/data/VOCdevkit/VOC2007/Annotations/{0}.xml'.format(
        #        str(i+1).zfill(7)))