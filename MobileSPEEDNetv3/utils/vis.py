import cv2
import matplotlib.pyplot as plt
import numpy as np

from spatialmath.base import q2r
from scipy.spatial.transform import Rotation as R
from matplotlib.pyplot import MultipleLocator

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (0, 0, 0) # White

def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    # x_min, y_min, w, h = bbox
    x_min, y_min, x_max, y_max = bbox

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img

def quat2dcm(q):

    """ Computing direction cosine matrix from quaternion, adapted from PyNav. """

    # normalizing quaternion
    q = q/np.linalg.norm(q)

    # q0 = q[0]
    # q1 = q[1]
    # q2 = q[2]
    # q3 = q[3]

    # dcm = np.zeros((3, 3))

    # dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    # dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    # dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

    # dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    # dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    # dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    # dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    # dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    # dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1
    dcm = q2r(q)

    return dcm


def project(q, r, camera):

        """ Projecting points to image frame to draw axes """

        # reference points in satellite frame for drawing axes
        p_axes = np.array([[0, 0, 0, 1],
                           [1, 0, 0, 1],
                           [0, 1, 0, 1],
                           [0, 0, 1, 1]])
        points_body = np.transpose(p_axes)

        # transformation to camera frame
        # pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
        rotation = R.from_quat([q[1], q[2], q[3], q[0]])
        pose_mat = np.hstack((rotation.as_matrix(), np.expand_dims(r, 1)))
        # pose_mat = np.hstack((quat2dcm(q), np.expand_dims(r, 1)))
        p_cam = pose_mat @ points_body

        # getting homogeneous coordinates
        points_camera_frame = p_cam / p_cam[2]

        # projection to image plane
        points_image_plane = camera.K @ points_camera_frame
        
        points_image_plane = camera.S @ points_image_plane

        x, y = (points_image_plane[0], points_image_plane[1])
        return x, y


def visualize_axes(ax, q, r, camera):

        """ Visualizing image, with ground truth pose with axes projected to training image. """

        if ax is None:
            ax = plt.gca()

        # no pose label for test
        xa, ya = project(q, r, camera)
        ax.arrow(xa[0], ya[0], xa[1] - xa[0], ya[1] - ya[0], head_width=20 * camera.S[0, 0], color='r')
        ax.arrow(xa[0], ya[0], xa[2] - xa[0], ya[2] - ya[0], head_width=20 * camera.S[0, 0], color='g')
        ax.arrow(xa[0], ya[0], xa[3] - xa[0], ya[3] - ya[0], head_width=20 * camera.S[0, 0], color='b')

        return

def visualize_image(image, bboxes, category_ids, category_id_to_name, ori, pos, camera):
    bboxes[0][0] = int(bboxes[0][0])
    bboxes[0][1] = int(bboxes[0][1])
    bboxes[0][2] = int(bboxes[0][2])
    bboxes[0][3] = int(bboxes[0][3])
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    visualize_axes(ax, np.array(ori), np.array(pos), camera)
    ax.imshow(img, cmap='gray')
    # 保存图片
    plt.savefig("test.png")
    # plt.show()

def visualize_encode(euler_range, euler_encode, stride=1):
    fig = plt.figure(figsize=(20, 8))
    # 以range为横坐标，euler_encode为纵坐标，绘制柱形图
    plt.bar(euler_range, euler_encode, width=stride)
    # 设置横坐标的标签
    plt.xticks(euler_range)
    # x标签旋转45度
    plt.xticks(rotation=45)
    # 设置纵坐标范围为0-1
    plt.ylim(0, 1)
    plt.xlim(euler_range[0], euler_range[-1])
    plt.tight_layout()
    # plt.show()
    plt.savefig("encode.png")