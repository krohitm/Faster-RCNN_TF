import _init_paths
import tensorflow as tf
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
plt.switch_backend('agg')
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2, csv
import argparse
from networks.factory import get_network
import pylab


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


#CLASSES = ('__background__','person','bike','motorbike','car','bus')

def vis_detections(im, class_name, dets,ax, images_dir, thresh=0.5):
    """Draw detected bounding boxes."""
    #print "in vis_detections"
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        #print 'True'
        return

    for i in inds:
        bbox = dets[i, :4]
        if class_name=='person':
            x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
            bounds = [x_min, y_min, x_max, y_max]
            #print "x_min: {0}, y_min: {1}, x_max: {2}, y_max: {3}".format(x_min,
            #              y_min, x_max, y_max)
            
            with open (os.path.join('/data0/krohitm/', 
                                    'object_boundaries/person.csv'),'a') as f:
                writer = csv.writer(f, delimiter = ',')
                writer.writerow(bounds)
        
        score = dets[i, -1]

        ax.add_patch(
           plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(sess, net, image_dir, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(image_dir, image_name)
    #im_file = os.path.join('/home/krohitm/code/Faster-RCNN_TF/data/temp_check/',image_name)
    im = cv2.imread(im_file)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        #print "in check"
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
	#print dets
        vis_detections(im, cls, dets, ax, images_dir, thresh=CONF_THRESH)
    pylab.savefig('/data0/krohitm/object_detection/out-{0}.jpg'.format((str(i)).zfill(7)), bbox_inches='tight')

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
    parser.add_argument('--images_dir', dest='images_dir', help='folder of images',
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
   
    #sess.run(tf.initialize_all_variables())

    print '\n\nLoaded network {:s}'.format(args.model)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(sess, net, im)

    #im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
    #            '001763.jpg', '004545.jpg']
    #im_names = ['1.jpg','2.jpg','3.jpg','4.jpg','5.jpg','6.jpg']

    #num_imgs = os.walk('ls -l {0} | wc -l'.format(images_dir))*2
    #num_imgs = os.system('ls -l $images_dir| wc -l') - 1
    _,_,files = os.walk("/home/krohitm/posture_dataset/100GOPRO/frames/train_val/").next()
    num_imgs = len(files)
    #print num_imgs
    with open (os.path.join('/data0/krohitm/','object_boundaries/person.csv'), 'w') as f:
                f.write('x_min,y_min,x_max,y_max\n')
    for i in range(1,num_imgs+1):
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Getting bounds for out-{0}.jpg'.format((str(i)).zfill(7))
        demo(sess, net, images_dir, 'out-{0}.jpg'.format((str(i)).zfill(7)))
    #plt.show()
