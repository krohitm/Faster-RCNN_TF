#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 16:57:45 2017

@author: krohitm
"""
import numpy as np
import os, csv,sys

median_boundaries = []

if __name__ == '__main__':
    dirpaths,dirnames,_ = os.walk('/data0/krohitm/posture_dataset/100GOPRO/frames/train_val'
                                  ).next()
    dirnames.sort()
    
    boundaries = np.array(np.loadtxt('/data0/krohitm/object_boundaries/person.csv',
                                     dtype = str, delimiter = ',', skiprows = 1, ))
    #print boundaries.shape
    all_dirs = np.apply_along_axis(lambda a: (a[0].split('/')),1,boundaries)
    all_dirs = all_dirs[:,-2]
    #print all_dirs.shape
    #print set(list(all_dirs))
    
    
    
    num_surrounding_frames = 20
    for dirpath, directory in zip(dirpaths, dirnames):
        current_directory = os.path.join(
        '/data0/krohitm/posture_dataset/100GOPRO/frames/train_val',directory)
        #print current_directory
        _,_,files = os.walk(current_directory).next()
        
        #for i in range
        num_imgs = list(all_dirs).count(directory)
        print "Getting median boundaries for the {0} images in {1} folder".format(
                num_imgs, directory)
        image_dir = os.path.join('/data0/krohitm/posture_dataset/100GOPRO/frames/train_val'
                                 ,directory)
        previous_set = 0
        for i in range(1,num_imgs+1):
            start = max(previous_set, previous_set+i-num_surrounding_frames)
            end = min(previous_set+num_imgs,i+2*num_surrounding_frames)-1
            #print start,end
            median_window = boundaries[start:end, 1:5].astype(float)
            #print median_window[0:5,:]
            bounds_for_median = np.median(median_window, axis = 0)
            if bounds_for_median.all() == -1:
                bounds_for_median = boundaries[i,:]
            image_name_full = os.path.join(
                    image_dir, '{0}.jpg'.format((str(i)).zfill(7)))
            #image_name_full = '{0}/{1}.jpg'.format(directory, (str(i)).zfill(7))
            bounds_for_median= np.array(map(str, bounds_for_median.tolist()))
            temp = bounds_for_median
            bounds_for_median = [image_name_full, temp[0], temp[1], temp[2], temp[3]]
            #bounds_for_median = np.insert(bounds_for_median,0,image_name_full, axis=0)
            #print bounds_for_median
            #break
            median_boundaries.append(bounds_for_median)
        #print image_name_full
        previous_set = i
    
    #median_boundaries = np.asarray(median_boundaries)
    
    with open (os.path.join('/data0/krohitm/','object_boundaries/person_median.csv'
                            ), 'w') as f:
        print "Writing bboxes to median file"
        f.write('image_name, x_min,y_min,x_max,y_max\n')
        writer = csv.writer(f, delimiter = ',')
        writer.writerows(median_boundaries)
    f.close()
    
    print "\a"