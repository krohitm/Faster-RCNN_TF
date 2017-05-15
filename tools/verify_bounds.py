import numpy as np
import os, csv, sys
import cv2
import matplotlib.pyplot as plt

median_boundaries = []

if __name__ == '__main__':
    dirpaths, dirnames, _ = os.walk('/data0/krohitm/posture_dataset/100GOPRO/frames/train_val'
                                    ).next()
    dirnames.sort()

    median_boundaries = np.array(np.loadtxt('/data0/krohitm/object_boundaries/person_median.csv',
                                     dtype=str, delimiter=',', skiprows=1, ))
    # print boundaries.shape
    all_dirs = np.apply_along_axis(lambda a: (a[0].split('/')), 1, median_boundaries)
    all_dirs = all_dirs[:, -2]

    #for i in range(len(median_boundaries[:,0])):
    for i in np.arange(49995,50005):
        image_name_full = median_boundaries[i,0]
        cur_image = image_name_full

        im = cv2.imread(cur_image)
        im = im[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')

        while cur_image == image_name_full:
            bbox = map(float, median_boundaries[i,1:5])
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=3.5)
            )
            plt.axis('off')
            plt.tight_layout()
            plt.draw()
            i += 1
            image_name_full = median_boundaries[i,0]
        plt.show()
        i = i-1
        #if i >=5:
        #    break

