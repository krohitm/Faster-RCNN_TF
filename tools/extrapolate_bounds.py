#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 20:17:02 2017

@author: krohitm
"""

import numpy as np
import os, csv

if __name__ == '__main__':
    boundaries = np.array(np.loadtxt('/Users/GodSpeed/Documents/ihealth_lab/person_extrapol.csv',
                                     dtype = str, delimiter = ',', skiprows = 1))
    discrepant_ranges = np.array(np.loadtxt('/Users/GodSpeed/Documents/ihealth_lab/discrepant_ranges_2.csv', 
                                          dtype = str, delimiter = ','))

    for disc_range in discrepant_ranges:
    	end_points = disc_range.split('-')
    	#print full_range
    	start_disc_frame = int(end_points[0])-1	#starting good frame
    	end_disc_frame = int(end_points[1])-1	#ending good frame
    	total_frames = end_disc_frame - start_disc_frame
    	per_frame_shift = (boundaries[end_disc_frame, 1:5].astype(float) - 
    		boundaries[start_disc_frame, 1:5].astype(float))/total_frames

    	for i in range(total_frames-1):
    		boundaries[start_disc_frame + i + 1, 1:5] = np.add(boundaries[
    			start_disc_frame + i,1:5].astype(float), per_frame_shift)

    with open (os.path.join(
    	'/Users/GodSpeed/Documents/ihealth_lab/person_extrapol_2.csv'), 'w') as f:
    	print "Writing bboxes to extrapolation file"
    	f.write('image_name, x_min,y_min,x_max,y_max\n')
    	writer = csv.writer(f, delimiter = ',')
    	writer.writerows(boundaries)
    f.close()