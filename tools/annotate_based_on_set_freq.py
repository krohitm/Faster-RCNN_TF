# Import the required modules
#import dlib
import cv2
import argparse
import get_pts_for_ims
import numpy as np
import os
import csv
#import matplotlib.pyplot as plt
#import re


def annotate(home_dir):
    #print "in annotate"
    all_image_names = []
    _,directories,_ = os.walk(home_dir).next()
    directories.sort()
    #print directories

    for directory in directories:
        #print directory
        full_path = os.path.join(home_dir, directory)
        _,_,files = os.walk(full_path).next()
        files.sort()
        full_file_names = ['{0}'.format(os.path.join(full_path,file_name)
            ) for file_name in files]
        all_image_names = all_image_names + full_file_names

    start = 0
    coords = []
    
    if os.path.isfile(os.path.join(home_dir, 'person_bbox.csv')):
        with open (os.path.join(home_dir, 'person_bbox.csv'), 'r') as f:
            print "Reading existing bbox file"
            reader = csv.reader(f, delimiter = ',')
            for row in reader:
                coords.append(row)
        f.close()
        del coords[0]
        #del coords[-1]
        
        i = 0
        start = len(coords)+1
    
    frames_to_skip = 5
    i = start
    #frame_counter = 0 #to skip frames and later extrapolate in between
    while i < len(all_image_names):
    #for i in range(start, len(all_image_names)):
        if len(all_image_names) - i < frames_to_skip:
            ext_frames = len(all_image_names) - i
        else:
            ext_frames = frames_to_skip
        if i%ext_frames != 0:    #drawing at a rate of 3fps for a 15 fps video
            i += 1
	    continue
        cur_fname=[all_image_names[i]]
        img=cv2.imread(cur_fname[0])
        points=get_pts_for_ims.run(img,np.array([0,0,0,0]), str(i))
        if points == [-5,-5,-5,-5]:
            print i-ext_frames
            points = cur_fname + coords[i-ext_frames][1:5]
            nothing_found = 0
        elif points == [-1,-1,-1,-1]:
            nothing_found = 1
            
            points = cur_fname + points
        else:
            points = cur_fname + points
            nothing_found = 0

        if max(map(float, points[1:5])) == 0:
            break

        #extrapolate for the middle four points
        if i != 0:
            if nothing_found == 1:
                for j in range(i-ext_frames+1, i):
                    coords.append([all_image_names[j]] + [-1,-1,-1,-1])
            else:
                per_frame_shift = (np.asarray(points[1:5], float) - np.asarray(
                        coords[i-ext_frames][1:5], float))/ext_frames
                for j in range(i-ext_frames+1, i):
                    points_temp = np.asarray(coords[j-1][1:5], float) + per_frame_shift
                    coords.append([all_image_names[j]] + list(points_temp))
        coords.append(points)
        print "Annotated till {}".format(cur_fname)
        i += 1
        
    with open (os.path.join(home_dir, 'person_bbox.csv'), 'w') as f:
        print "Writing bboxes to bbox file"
        f.write('image_name,x_min,y_min,x_max,y_max\n')
        writer = csv.writer(f, delimiter = ',')
        writer.writerows(coords)
    f.close()


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--parent_folder", required=True, help="Path to parent folder")

    args = vars(ap.parse_args())
    parent_folder = args['parent_folder']
    annotate(parent_folder)
