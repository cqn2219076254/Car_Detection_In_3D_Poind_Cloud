import numpy as np
import os

CALIB = './data/calib'
AFT_DATA = './data/aft_data'
LABEL = './data/label'


def read_result(n):
    file = open(os.path.join(AFT_DATA, n + '.txt'))
    result = []
    for _ in range(100):
        data = file.readline().split(' ')[0:8]
        result.append(np.array(data, dtype=np.float32))
    result = np.array(result)
    box = result[0:100, 0:7]
    score = result[0:100, 7:].reshape(100)
    return box, score


def get_ca_lib(filename):
    txt = open(filename)
    data = txt.readlines()
    lib1 = np.array(data[2].split(' ')[1:], dtype=np.float32).reshape(3, 4)
    return lib1


def get_box3d_corners_helper_np(centers, headings, sizes):
    N = centers.shape[0]
    l = sizes[:, 0]
    h = sizes[:, 1]
    w = sizes[:, 2]

    z = np.zeros_like(l)
    x_corners = np.stack([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], axis=1)  # (N,8)
    y_corners = np.stack([z, z, z, z, -h, -h, -h, -h], axis=1)  # (N,8)
    z_corners = np.stack([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], axis=1)  # (N,8)
    corners = np.concatenate([np.expand_dims(x_corners, 1), np.expand_dims(y_corners, 1), np.expand_dims(z_corners, 1)],
                             axis=1)  # (N,3,8)
    # print x_corners, y_corners, z_corners
    c = np.cos(headings)
    s = np.sin(headings)
    ones = np.ones([N], dtype=np.float32)
    zeros = np.zeros([N], dtype=np.float32)
    row1 = np.stack([c, zeros, s], axis=1)  # (N,3)
    row2 = np.stack([zeros, ones, zeros], axis=1)
    row3 = np.stack([-s, zeros, c], axis=1)
    R = np.concatenate([np.expand_dims(row1, 1), np.expand_dims(row2, 1), np.expand_dims(row3, 1)], axis=1)  # (N,3,3)
    # print row1, row2, row3, R, N
    corners_3d = np.matmul(R, corners)  # (N,3,8)
    corners_3d += np.tile(np.expand_dims(centers, 2), [1, 1, 8])  # (N,3,8)
    corners_3d = np.transpose(corners_3d, [0, 2, 1])  # (N,8,3)
    return corners_3d


def project_to_image(pts_3d, P):
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]


def project_to_image_space_corners(anchors_corners, stereo_calib_p2, img_shape=(375, 1242)):
    if anchors_corners.shape[1] != 8:
        raise ValueError("Invalid shape for anchors {}, should be "
                         "(N, 8, 3)")

    # Apply the 2D image plane transformation
    anchors_corners = np.reshape(anchors_corners, [-1, 3])
    pts_2d = project_to_image(anchors_corners, stereo_calib_p2)  # [-1, 2]

    pts_2d = np.reshape(pts_2d, [-1, 8, 2])

    h, w = img_shape

    # Get the min and maxes of image coordinates
    i_axis_min_points = np.minimum(np.maximum(np.amin(pts_2d[:, :, 0], axis=1), 0), w)
    j_axis_min_points = np.minimum(np.maximum(np.amin(pts_2d[:, :, 1], axis=1), 0), h)

    i_axis_max_points = np.minimum(np.maximum(np.amax(pts_2d[:, :, 0], axis=1), 0), w)
    j_axis_max_points = np.minimum(np.maximum(np.amax(pts_2d[:, :, 1], axis=1), 0), h)

    box_corners = np.stack([i_axis_min_points, j_axis_min_points,
                            i_axis_max_points, j_axis_max_points], axis=-1)
    return np.array(box_corners, dtype=np.float32)


def main(n):
    pred_bbox_3d_op, pred_cls_score_op = read_result(n)
    calib_P = get_ca_lib(os.path.join(CALIB, n + '.txt'))
    select_idx = np.where(pred_cls_score_op >= 0.3)[0]
    pred_cls_score_op = pred_cls_score_op[select_idx]
    pred_bbox_3d_op = pred_bbox_3d_op[select_idx]
    pred_bbox_corners_op = get_box3d_corners_helper_np(pred_bbox_3d_op[:, :3], pred_bbox_3d_op[:, -1],
                                                       pred_bbox_3d_op[:, 3:-1])
    pred_bbox_2d = project_to_image_space_corners(pred_bbox_corners_op, calib_P)
    pred_bbox_3d_op = pred_bbox_3d_op[:, [4, 5, 3, 0, 1, 2, 6]]
    pred_cls_score_op = pred_cls_score_op.reshape((len(pred_cls_score_op), 1))
    pred = np.hstack((pred_bbox_2d, pred_bbox_3d_op))
    pred = np.hstack((pred, pred_cls_score_op))
    return pred


if __name__ == "__main__":
    f = open('./data/list.txt')
    lists = f.readlines()
    for i in lists:
        i = i.strip('\n')
        print(i)
        aft_data = main(i)
        with open(os.path.join(LABEL, i + '.txt'), 'w') as f:
            for _ in aft_data:
                f.write('Car 0.00 0 -10 ')
                for j in _:
                    f.write(str(j) + ' ')
                f.write('\n')
    print("process finished.")