import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.switch_backend('agg') 

median_boundaries = []

def quit_figure(event):
    if event.key == 'n':
        plt.close(event.canvas.figure)

def close_event():
    plt.close()

vid_array = []

if __name__ == '__main__':
    median_boundaries = np.array(np.loadtxt('/data0/krohitm/object_boundaries/person_extrapol.csv',
                                     dtype=str, delimiter=',', skiprows=1, ))
        
    
    for i in range(140000,len(median_boundaries[:,0])):
    #for i in np.arange(49995,50005):
        #image_name_full = median_boundaries[i,0]
        image_name_full = '/home/krohitm/code/Faster-RCNN_TF/data/VOCdevkit2007/VOC2007/JPEGImages/{0}.jpg'.format(
                ((str(i+1)).zfill(7)))
                
        #print image_name_full
        cur_image = image_name_full

        im = cv2.imread(cur_image)
        height, width, layers = im.shape
        
        im = im[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')

        #while cur_image == image_name_full:
        bbox = map(float, median_boundaries[i,1:5])
        ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=3.5))
        plt.axis('off')
        plt.tight_layout()
        plt.draw()
        #i += 1
        #image_name_full = median_boundaries[i,0]
        plt.title(cur_image)
        plt.savefig('/data0/krohitm/object_detection/{0}.jpg'.format((
                str(i+1)).zfill(7)), bbox_inches='tight')
        print "Saved {0}.jpg".format(str(i+1)).zfill(7)
        plt.close()
        
        #timer = fig.canvas.new_timer(interval = 3000) #creating a timer object and setting an interval of 3000 milliseconds
        #timer.add_callback(close_event)
        #cid = plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)
        #timer.start()
        #plt.show()
        #i = i-1
