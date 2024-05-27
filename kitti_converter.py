import os
import glob
import argparse
import numpy as np

import scipy.io as sio


class LabelToKittiConverter:
    """ KITTI FORMAT
    #Values    Name      Description
    ----------------------------------------------------------------------------
       1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                         'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                         'Misc' or 'DontCare'
       1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                         truncated refers to the object leaving image boundaries
       1    occluded     Integer (0,1,2,3) indicating occlusion state:
                         0 = fully visible, 1 = partly occluded
                         2 = largely occluded, 3 = unknown
       1    alpha        Observation angle of object, ranging [-pi..pi]
       4    bbox         2D bounding box of object in the image (0-based index):
                         contains left, top, right, bottom pixel coordinates
       3    dimensions   3D object dimensions: height, width, length (in meters)
       3    location     3D object location x,y,z in camera coordinates (in meters)
       1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
       1    score        Only for results: Float, indicating confidence in
                         detection, needed for p/r curves, higher is better.
    """
    def __init__(self, conversion, gt_labels_fp, save_path="/media/simu/8TB_HDD/ThreeD_detection/data/site_data/dataset/training"):
        conversion_dict = self.get_attribute_idx(conversion)
        
        
        annotations = []
        with open(gt_labels_fp, "r") as f:
            lines = f.readlines()
            for l in lines:
                anno = [float(x) for x in l.split("\n")[0].split(" ")]
                annotations.append(anno)
        print(annotations)
        self.annotations = annotations
        if gt_labels_fp is not None:
            print("------ Ground Truth Labels Conversion -------")
            new_labels_fp = os.path.join(save_path,  'label_2/')
            os.makedirs(new_labels_fp, exist_ok=True)
            self.convert_to_kitti(conversion_dict, new_labels_fp, pred_labels=False)

    def convert_to_kitti(self, conversion_key, new_fp, pred_labels=False):
        for num, example in enumerate(self.annotations):
            new_label = self.new_label_from_txt(example, conversion_key, pred=pred_labels)
            np.savetxt(new_fp + str(num) + ".txt", new_label, delimiter=' ',
                           fmt='%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s')
            print(new_fp + str(num) + ".txt")

    def new_label_from_txt(self, old_label, idx_key, pred=True):
        classes = []
        trunc = []
        occ = []
        obs = []
        camera_box = []
        score = []
        x, y, z, r = [], [], [], []
        l, w, h = [], [], []

        classes.append(['Pedestrian']) # assume if no class is specified, its a car
        trunc.append(0.00)
        occ.append(0)
        obs.append(0)
        camera_box.append((0, 0, 50, 50))
        x.append(old_label[0])
        y.append(old_label[1])
        z.append(old_label[2] - old_label[5]/2)
        r.append(old_label[-2])
        l.append(old_label[3])
        w.append(old_label[4])
        h.append(old_label[5])
        final_array = np.hstack((
            np.array(classes).reshape(-1, 1),
            np.array(trunc).reshape(-1, 1),
            np.array(occ).reshape(-1,1 ),
            np.array(obs).reshape(-1, 1),
            np.array(camera_box),
            np.array(h).reshape(-1, 1),
            np.array(w).reshape(-1, 1),
            np.array(l).reshape(-1, 1),
            np.array(x).reshape(-1, 1),
            np.array(y).reshape(-1, 1),
            np.array(z).reshape(-1, 1),
            np.array(r).reshape(-1, 1)
        ))
        if pred:
            final_array = np.hstack((final_array, np.array(score).reshape(-1, 1)))
        return final_array

    @staticmethod
    def get_attribute_idx(conversion):
        idx_dict = {}
        conversion = conversion.split(" ")
        print("-- Your conversion key --")
        for i, attribute in enumerate(conversion):
            print(i, attribute)
            if attribute == 'class':
                idx_dict['class'] = i
            elif attribute == 'truncated':
                idx_dict['truncated'] = i
            elif attribute == 'occluded':
                idx_dict['occluded'] = i
            elif attribute == 'alpha':
                idx_dict['alpha'] = i
            elif attribute == 'x1':
                idx_dict['x1'] = i # assumes [x1 y1 x2 y2] follows
            elif attribute == 'x':
                idx_dict['x'] = i
            elif attribute == 'y':
                idx_dict['y'] = i
            elif attribute == 'z':
                idx_dict['z'] = i
            elif attribute == 'l':
                idx_dict['l'] = i
            elif attribute == 'w':
                idx_dict['w'] = i
            elif attribute == 'h':
                idx_dict['h'] = i
            elif attribute == 'r':
                idx_dict['r'] = i
            elif attribute == 'score':
                idx_dict['score'] = i
        return idx_dict

def read_mat_file(path):
    test = sio.loadmat(path)
    print(test['__function_workspace__'].shape)

def main():
    # parser = argparse.ArgumentParser(description='arg parser')
    # parser.add_argument('--pred_labels', type=str, help='file path with your prediction labels')
    # parser.add_argument('--gt_labels', type=str, help='file path with your ground truth labels')
    # parser.add_argument('--format', type=str, help='your label format e.g. "class, x, y, z, l, w, h"', required=True)
    # parser.add_argument('--csv', dest='csv', action='store_true', help='is your file csv or space separated')
    # args = parser.parse_args()
    # print("------ Converting to KITTI Labels -------")


    LabelToKittiConverter(conversion="class, x, y, z, l, w, h",
                          gt_labels_fp="/home/simu/code/ThreeD_detection/data/site_data/point_cloud_data_700_800_labeled.txt")

    # read_mat_file("/home/simu/code/ThreeD_detection/data/site_data/point_cloud_data_700_800_labeled.mat")


if __name__ == '__main__':
    main()