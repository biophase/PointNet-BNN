import numpy as np
import sys
import os
print(sys.path[0])
pred_data_label_filenames = [line.rstrip() for line in open(os.path.join(sys.path[0],'all_pred_data_label_filelist.txt'))]
gt_label_filenames = [f.rstrip('_pred\.txt') + '_gt.txt' for f in pred_data_label_filenames]
num_room = len(gt_label_filenames)


gt_classes = [0 for _ in range(14)]
positive_classes = [0 for _ in range(14)]
true_positive_classes = [0 for _ in range(14)]
for i in range(num_room):
    print(i)
    data_label = np.loadtxt(pred_data_label_filenames[i])
    pred_label = data_label[:,-1]
    gt_label = np.loadtxt(gt_label_filenames[i])
    print(gt_label.shape)
    for j in range(gt_label.shape[0]):
        gt_l = int(gt_label[j])
        pred_l = int(pred_label[j])
        gt_classes[gt_l] += 1
        positive_classes[pred_l] += 1
        true_positive_classes[gt_l] += int(gt_l==pred_l)


print(gt_classes)
print(positive_classes)
print(true_positive_classes)


print('Overall accuracy: {0}'.format(sum(true_positive_classes)/float(sum(positive_classes))))

print ('IoU:')
iou_list = []
for i in range(14):
    iou = true_positive_classes[i]/float(gt_classes[i]+positive_classes[i]-true_positive_classes[i]) 
    print(iou)
    iou_list.append(iou)

print(sum(iou_list)/14.0)
