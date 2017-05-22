import _init_paths
import tensorflow as tf
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
#import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import numpy as np
import os, cv2, csv
import argparse
from networks.factory import get_network


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

boundaries = []
median_boundaries = []

def demo(sess, net, image_dir, image_name):
    """Detect object classes in an image using pre-computed object proposals."""
    #has_class = 0
    # Load the demo image
    im_file = os.path.join(image_dir, image_name)
    
    im = cv2.imread(im_file)
    # Detect all object classes and regress object bounds
    timer.tic()
    scores, boxes = im_detect(sess, net, im)

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    
    #only using class_ind 15 for class person
    for cls_ind, cls in enumerate(CLASSES[15:16]):
        #print "in check"
        cls_ind += 15 # because we skipped all other classes
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) == 0:
            boundaries.append(np.array([os.path.join(image_dir, image_name
                                                 ), -1, -1, -1, -1]))
            return

        #temp_boundaries = [0,0,0,0]
        for i in inds:
            bbox = dets[i, :4]
            x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
            boundaries.append(np.array(
                        [os.path.join(image_dir, image_name), x_min, y_min, 
                         x_max, y_max]))
    
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='VGGnet_test')
    parser.add_argument('--model', dest='model', help='Model path',
                        default=' ')
    parser.add_argument('--images_dir', dest='images_dir', 
                        help='parent folder of images',
                        default='None')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
    images_dir = args.images_dir
    
    if args.model == ' ':
        raise IOError(('Error: Model not found.\n'))
        
    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load network
    net = get_network(args.demo_net)
    # load model
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    saver.restore(sess, args.model)
   
    print '\n\nLoaded network {:s}'.format(args.model)
    
    dirpaths,dirnames,_ = os.walk('/data0/krohitm/posture_dataset/100GOPRO/frames/train_val'
                                  ).next()
    dirnames.sort()
    
    image_counter = 1
    for dirpath, directory in zip(dirpaths, dirnames):
        image_dir = os.path.join('/data0/krohitm/posture_dataset/100GOPRO/frames/train_val'
                                 ,directory)
        _,_,files = os.walk(image_dir).next()
        num_imgs = len(files)
        
        for i in range(1,num_imgs+1):
            timer = Timer()
            print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
            print 'Getting bounds for {0}/{1}.jpg'.format(directory, (
                    str(i)).zfill(7))
            timer.tic()
            
            demo(sess, net, image_dir, '{0}.jpg'.format((str(i)).zfill(7)))
            
            timer.toc()
            print ("detection took {:.3f}s".format(timer.total_time))
            image_counter+=1
    
    boundaries = np.asarray(boundaries)
    
    with open (os.path.join('/data0/krohitm/','object_boundaries/person.csv'
                            ), 'w') as f:
        print "Writing bboxes to actual identified bboxes file"
        f.write('image_name, x_min,y_min,x_max,y_max\n')
        writer = csv.writer(f, delimiter = ',')
        writer.writerows(boundaries)
    f.close()
    