#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 16:57:45 2017

@author: krohitm
"""
import numpy as np
import os, csv

median_boundaries = []

if __name__ == '__main__':
    dirpaths,dirnames,_ = os.walk('/data0/krohitm/posture_dataset/100GOPRO/frames/train_val'
                                  ).next()
    dirnames.sort()
    
    boundaries = np.array(np.loadtxt('/data0/krohitm/object_boundaries/person.csv',
                                     dtype = str, delimiter = ',', skiprows = 1, ))
    
    all_dirs = np.apply_along_axis(lambda a: (a[0].split('/')),1,boundaries)
    all_dirs = all_dirs[:,-2]
    
    
    previous_set = 0
    num_surrounding_frames = 10
    for dirpath, directory in zip(dirpaths, dirnames):
        current_directory = os.path.join(
        '/data0/krohitm/posture_dataset/100GOPRO/frames/train_val',directory)
        
        _,_,files = os.walk(current_directory).next()
        
        
        num_imgs = list(all_dirs).count(directory)
        print "Getting median boundaries for the {0} images in {1} folder".format(
                num_imgs, directory)
        image_dir = os.path.join('/data0/krohitm/posture_dataset/100GOPRO/frames/train_val'
                                 ,directory)
        for i in range(num_imgs):
            start = max(previous_set, previous_set+i-num_surrounding_frames)
            end = min(previous_set+num_imgs, previous_set+i+num_surrounding_frames)+1
            median_window = boundaries[start:end, 1:5].astype(float)
            bounds_for_median = np.median(median_window, axis = 0)
            
            if np.array_equal(bounds_for_median, np.array([-1,-1,-1,-1])):
                bounds_for_median = np.array(median_boundaries[i-1][1:5])
            
	    #should have been using the exact image names instead
            image_name_full = os.path.join(
                    image_dir, '{0}.jpg'.format((str(i+1)).zfill(7)))
            bounds_for_median= np.array(map(str, bounds_for_median.tolist()))
            bbox = bounds_for_median
            
            median_boundaries.append(np.array([image_name_full, bbox[0], bbox[1],
                                               bbox[2], bbox[3]]))
        previous_set += i
        
    
    with open (os.path.join('/data0/krohitm/','object_boundaries/person_median.csv'
                            ), 'w') as f:
        print "Writing bboxes to median file"
        f.write('image_name, x_min,y_min,x_max,y_max\n')
        writer = csv.writer(f, delimiter = ',')
        writer.writerows(median_boundaries)
    f.close()
