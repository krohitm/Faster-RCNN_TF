#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 20:17:02 2017

@author: krohitm
"""

import numpy as np
import os, csv

if __name__ == '__main__':
    median_boundaries = np.array(np.loadtxt('/data0/krohitm/object_boundaries/person_median.csv',
                                     dtype = str, delimiter = ',', skiprows = 1, ))
    discrepant_bbox = np.array(np.loadtxt('/data0/krohitm/object_boundaries/discrepant_ranges.csv', 
                                          dtypr = str, delimter = ',', skiprows = 1))
    directories = len(median_boundaries[:,0])
    
    num_imgs = len(median_boundaries[:,0])
    num_discrepent_ranges = len(discrepant_bbox[:,0])
    
    image_home_dir = '/data0/krohitm/posture_dataset/100GOPRO/frames/train_val/'
    
    directories.sort()
    for i in range(len(directories)):
        discrepent_ranges = discrepant_bbox[i,0].split(',')
        num_discrepent_ranges = len(discrepent_ranges)
        
        for disc_range in discrepent_ranges:
            #folder_name = disc_range[0]
            folder_name = directories[i]
            
            start_disc_frame = os.path.join(image_home_dir, folder_name, 
                                            ((disc_range[1].split('-')[0]).zfill(7) + '.jpg'))
            if len(disc_range[1].split('-') == 1):
                end_disc_image = start_disc_frame
            else:
                end_disc_frame = os.path.join(image_home_dir, folder_name, 
                                              ((disc_range[1].split('-')[1]).zfill(7) + '.jpg'))
        
            first_frame = np.where(median_boundaries[:,0] == start_disc_frame)
            last_frame = np.where(median_boundaries[:,0] == end_disc_frame)
            total_frames = last_frame - first_frame + 2 #+2 for the surrounding correct frames
                
            per_frame_shift = (map(float, median_boundaries[last_frame + 1, 1:5]) - map(
                    float, median_boundaries[first_frame - 1, 1:5]))/total_frames
        
            for i in range(total_frames-1):
                median_boundaries[first_frame+i] = median_boundaries[first_frame+i-1] + per_frame_shift
    
    with open (os.path.join('/data0/krohitm/','object_boundaries/person_extrapol.csv'
                            ), 'w') as f:
        print "Writing bboxes to extrapolation file"
        f.write('image_name, x_min,y_min,x_max,y_max\n')
        writer = csv.writer(f, delimiter = ',')
        writer.writerows(median_boundaries)
    f.close()
