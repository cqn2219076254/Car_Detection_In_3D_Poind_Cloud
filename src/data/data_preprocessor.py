import numpy as np
import os
import cv2

MAX_POINT_NUMBER = 16384
POINT = './data/point'
CALIB = './data/calib'
IMAGE = './data/image'
PRE_DATA = './data/pre_data'


def get_ca_lib(filename):
    txt = open(filename)
    lib = txt.readlines()
    lib1 = np.array(lib[2].split(' ')[1:], dtype=np.float32).reshape(3, 4)
    lib2 = np.array(lib[4].split(' ')[1:], dtype=np.float32).reshape(3, 3)
    lib3 = np.array(lib[5].split(' ')[1:], dtype=np.float32).reshape(3, 4)
    return lib1, lib2, lib3


def get_points(filename):
    scan = np.fromfile(filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    set1 = scan[:, :3]
    set2 = scan[:, 3:]
    return set1, set2


def main(n):
    points, intensity = get_points(os.path.join(POINT, n + '.bin'))
    p, r0, v2c = get_ca_lib(os.path.join(CALIB, n + '.txt'))
    img = cv2.imread(os.path.join(IMAGE, n + '.png'))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # project_velo_to_rect
    n = points.shape[0]
    points = np.hstack((points, np.ones((n, 1))))
    points = np.dot(points, np.transpose(v2c))
    points = np.transpose(np.dot(r0, np.transpose(points)))
    # get_point_filter_in_image
    image_shape = img.shape
    img_coord = np.hstack((points, np.ones((n, 1))))
    img_coord = np.dot(img_coord, np.transpose(p))
    img_coord[:, 0] /= img_coord[:, 2]
    img_coord[:, 1] /= img_coord[:, 2]
    img_coord = img_coord[:, 0:2]
    point_filter = ((img_coord[:, 0] >= 0) &
                    (img_coord[:, 0] < image_shape[1]) &
                    (img_coord[:, 1] >= 0) &
                    (img_coord[:, 1] < image_shape[0]))
    z_filter = points[:, 2] >= 0
    point_filter = np.logical_and(point_filter, z_filter)
    # get_point_filter
    extents = [[-40, 40], [-5, 3], [0, 70]]
    extents = np.array(extents)
    extents_filter = (points[:, 0] > extents[0][0]) & \
                     (points[:, 0] < extents[0][1]) & \
                     (points[:, 1] > extents[1][0]) & \
                     (points[:, 1] < extents[1][1]) & \
                     (points[:, 2] > extents[2][0]) & \
                     (points[:, 2] < extents[2][1])
    point_filter = np.logical_and(point_filter, extents_filter)
    point_filter = np.where(point_filter)[0]
    point_num = len(point_filter)
    if point_num > MAX_POINT_NUMBER:
        sampled_idx = np.random.choice(point_filter, MAX_POINT_NUMBER, replace=False)
    else:
        sampled_idx_1 = np.random.choice(point_filter, point_num, replace=False)
        sampled_idx_2 = np.random.choice(point_filter, MAX_POINT_NUMBER - point_num, replace=True)
        sampled_idx = np.concatenate([sampled_idx_1, sampled_idx_2], axis=0)
    points = points[sampled_idx]
    intensity = intensity[sampled_idx]
    p_data = np.hstack((points, intensity))
    return p_data


if __name__ == "__main__":
    f = open('./data/list.txt')
    lists = f.readlines()
    for i in lists:
        i = i.strip('\n')
        print(i)
        pre_data = main(i)
        with open(os.path.join(PRE_DATA, i + '.txt'), 'w') as f:
            for _ in pre_data:
                for j in _:
                    f.write(str(j) + " ")
                f.write('\n')
    print("preprocess finished.")
