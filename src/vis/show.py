import argparse
import os
import kitti_util as utils
import numpy as np
import cv2
import mayavi.mlab as mlab
from viz_util import draw_lidar, draw_gt_boxes3d
raw_input = input


class Object(object):
    def __init__(self, root_dir, split="data"):
        """root_dir contains test and testing folders"""
        self.root_dir = root_dir
        self.split = split
        print(root_dir, split)
        self.split_dir = os.path.join(root_dir, split)

        if split == "data":
            self.num_samples = 7481

        lidar_dir = "point"
        depth_dir = "depth"
        pred_dir = "pred"

        self.image_dir = os.path.join(self.split_dir, "image")
        self.label_dir = os.path.join(self.split_dir, "label")
        self.calib_dir = os.path.join(self.split_dir, "calib")

        self.depthpc_dir = os.path.join(self.split_dir, "depth_pc")
        self.lidar_dir = os.path.join(self.split_dir, lidar_dir)
        self.depth_dir = os.path.join(self.split_dir, depth_dir)
        self.pred_dir = os.path.join(self.split_dir, pred_dir)

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert idx < self.num_samples
        img_filename = os.path.join(self.image_dir, "%06d.png" % idx)
        return utils.load_image(img_filename)

    def get_lidar(self, idx, dtype=np.float32, n_vec=4):
        assert idx < self.num_samples
        lidar_filename = os.path.join(self.lidar_dir, "%06d.bin" % idx)
        print(lidar_filename)
        return utils.load_velo_scan(lidar_filename, dtype, n_vec)

    def get_calibration(self, idx):
        assert idx < self.num_samples
        calib_filename = os.path.join(self.calib_dir, "%06d.txt" % idx)
        return utils.Calibration(calib_filename)

    def get_label_objects(self, idx):
        assert idx < self.num_samples and self.split == "data"
        label_filename = os.path.join(self.label_dir, "%06d.txt" % idx)
        return utils.read_label(label_filename)

    def get_pred_objects(self, idx):
        assert idx < self.num_samples
        pred_filename = os.path.join(self.pred_dir, "%06d.txt" % idx)
        is_exist = os.path.exists(pred_filename)
        if is_exist:
            return utils.read_label(pred_filename)
        else:
            return None

    def get_depth(self, idx):
        assert idx < self.num_samples
        img_filename = os.path.join(self.depth_dir, "%06d.png" % idx)
        return utils.load_depth(img_filename)


def box_min_max(box3d):
    box_min = np.min(box3d, axis=0)
    box_max = np.max(box3d, axis=0)
    return box_min, box_max


def get_velo_whl(box3d, pc):
    bmin, bmax = box_min_max(box3d)
    ind = np.where(
        (pc[:, 0] >= bmin[0])
        & (pc[:, 0] <= bmax[0])
        & (pc[:, 1] >= bmin[1])
        & (pc[:, 1] <= bmax[1])
        & (pc[:, 2] >= bmin[2])
        & (pc[:, 2] <= bmax[2])
    )[0]
    # print(pc[ind,:])
    if len(ind) > 0:
        vmin, vmax = box_min_max(pc[ind, :])
        return vmax - vmin
    else:
        return 0, 0, 0, 0


def stat_lidar_with_boxes(pc_velo, objects, calib):
    """ Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) """
    for obj in objects:
        if obj.type == "DontCare":
            continue
        _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        v_l, v_w, v_h, _ = get_velo_whl(box3d_pts_3d_velo, pc_velo)
        print("%.4f %.4f %.4f %s" % (v_w, v_h, v_l, obj.type))


def get_lidar_in_image_fov(
    pc_velo, calib, xmin, ymin, xmax, ymax, return_more=False, clip_distance=2.0
):
    """ Filter lidar points, keep those in image FOV """
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (
        (pts_2d[:, 0] < xmax)
        & (pts_2d[:, 0] >= xmin)
        & (pts_2d[:, 1] < ymax)
        & (pts_2d[:, 1] >= ymin)
    )
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def get_lidar_index_in_image_fov(
    pc_velo, calib, xmin, ymin, xmax, ymax, clip_distance=2.0
):
    """ Filter lidar points, keep those in image FOV """
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (
        (pts_2d[:, 0] < xmax)
        & (pts_2d[:, 0] >= xmin)
        & (pts_2d[:, 1] < ymax)
        & (pts_2d[:, 1] >= ymin)
    )
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    return fov_inds


def show_lidar_with_depth(
        pc_velo,
        objects,
        p_objects,
        calib,
        fig,
        img_width=None,
        img_height=None,
        pc_label=False,
):
    """ Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) """

    print(("All point num: ", pc_velo.shape[0]))
    pc_velo_index = get_lidar_index_in_image_fov(
        pc_velo[:, :3], calib, 0, 0, img_width, img_height
    )
    pc_velo = pc_velo[pc_velo_index, :]
    print(("FOV point num: ", pc_velo.shape))
    print("pc_velo", pc_velo.shape)
    draw_lidar(pc_velo, fig=fig, pc_label=pc_label)

    for obj in objects:
        if obj.type == "DontCare":
            continue
        # Draw 3d bounding box
        _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        print("box3d_pts_3d_velo:")
        print(box3d_pts_3d_velo)

        # TODO: change the color of boxes
        if obj.type == "Car":
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=(0, 1, 0), label=obj.type)
        elif obj.type == "Pedestrian":
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=(0, 1, 1), label=obj.type)
        elif obj.type == "Cyclist":
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=(1, 1, 0), label=obj.type)

    if p_objects is not None:
        color = (1, 0, 0)
        for obj in p_objects:
            if obj.type == "Car":
                # Draw 3d bounding box
                _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
                box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
                print("box3d_pts_3d_velo:")
                print(box3d_pts_3d_velo)
                draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)
                # Draw heading arrow
                _, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
                ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
                x1, y1, z1 = ori3d_pts_3d_velo[0, :]
                x2, y2, z2 = ori3d_pts_3d_velo[1, :]
                mlab.plot3d(
                    [x1, x2],
                    [y1, y2],
                    [z1, z2],
                    color=color,
                    tube_radius=None,
                    line_width=1,
                    figure=fig,
                )

    mlab.show(1)


def show_image_with_boxes(img, objects, p_objects, calib, depth=None):
    """ Show image with 2D bounding boxes """
    img1 = np.copy(img)  # for 2d bbox
    img2 = np.copy(img)  # for 3d bbox
    for obj in objects:
        if obj.type == "DontCare":
            continue
        if obj.type == "Car":
            cv2.rectangle(
                img1,
                (int(obj.xmin), int(obj.ymin)),
                (int(obj.xmax), int(obj.ymax)),
                (0, 255, 0),
                2,
            )
        if obj.type == "Pedestrian":
            cv2.rectangle(
                img1,
                (int(obj.xmin), int(obj.ymin)),
                (int(obj.xmax), int(obj.ymax)),
                (255, 255, 0),
                2,
            )
        if obj.type == "Cyclist":
            cv2.rectangle(
                img1,
                (int(obj.xmin), int(obj.ymin)),
                (int(obj.xmax), int(obj.ymax)),
                (0, 255, 255),
                2,
            )
        box3d_pts_2d, _ = utils.compute_box_3d(obj, calib.P)
        if obj.type == "Car":
            img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(0, 255, 0))
        elif obj.type == "Pedestrian":
            img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(255, 255, 0))
        elif obj.type == "Cyclist":
            img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(0, 255, 255))

    cv2.imshow("2dbox", img1)
    show3d = True
    if show3d:
        cv2.imshow("3dbox", img2)
    if depth is not None:
        cv2.imshow("depth", depth)

    return img1, img2


def dataset_viz(root_dir, args):
    dataset = Object(root_dir, split="data")
    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
    )
    for data_idx in range(len(dataset)):
        if args.ind > 0:
            data_idx = args.ind
        # Load data from dataset
        objects = dataset.get_label_objects(data_idx)
        p_objects = None
        if args.pred:
            p_objects = dataset.get_pred_objects(data_idx)
        n_vec = 4

        dtype = np.float32
        pc_velo = dataset.get_lidar(data_idx, dtype, n_vec)[:, 0:n_vec]
        calib = dataset.get_calibration(data_idx)
        img = dataset.get_image(data_idx)
        img_height, img_width, _ = img.shape
        print(data_idx, "image shape: ", img.shape)
        print(data_idx, "velo  shape: ", pc_velo.shape)
        depth = None
        print("======== Objects in Ground Truth ========")
        n_obj = 0
        for obj in objects:
            if obj.type != "DontCare":
                print("=== {} object ===".format(n_obj + 1))
                obj.print_object()
                n_obj += 1
        if args.pred:
            for obj in p_objects:
                if obj.type != "DontCare":
                    print("=== {} object ===".format(n_obj + 1))
                    obj.print_object()
                    n_obj += 1

        # show_image_with_boxes_3type(img, objects, calib, objects2d, data_idx, objects_pred)
        show_image_with_boxes(img, objects, p_objects, calib, depth)
        # Draw 3d box in LiDAR point cloud
        show_lidar_with_depth(
            pc_velo,
            objects,
            p_objects,
            calib,
            fig,
            img_width,
            img_height,
            pc_label=False,
        )
        input_str = raw_input()

        mlab.clf()
        if input_str == "killall":
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Object Visualization")
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        default="",
        metavar="Path",
        help="input dir",
    )
    parser.add_argument(
        "-i",
        "--ind",
        type=int,
        metavar="N",
        help="input index",
    )
    parser.add_argument(
        "-p",
        "--pred",
        action="store_true",
        help="show predict results"
    )
    args = parser.parse_args()
    dataset_viz(args.dir, args)
