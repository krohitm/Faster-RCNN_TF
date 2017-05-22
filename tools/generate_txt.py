import numpy as np
import csv

person_train_file_num = map(lambda q: (q.zfill(7)), map(str, np.arange(1,237968)))
person_train_exists = map(str, np.ones(237967, dtype = np.int))
#print person_train_file_num
#print person_train_exists

print np.column_stack((person_train_file_num,person_train_exists))
with open ('/home/krohitm/code/Faster-RCNN_TF/data/VOCdevkit2007/VOC2007/ImageSets/Main/train.txt','w') as f:
        print "Storing person_train.txt"
        #f.write('image_name, x_min,y_min,x_max,y_max\n')
        writer = csv.writer(f, delimiter = ' ')
        writer.writerows(np.column_stack((person_train_file_num, person_train_exists)))
f.close()
