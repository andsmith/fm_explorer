import time
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def add_text(img, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255),
             bkg_color=None, bkg_margin=5, thickness=None, linestyle=cv2.LINE_AA, y_axis_flip=False, v_align='bottom',
             h_align='left', dry_run=False):
    """
    Most args are for cv2 puttext.
    :param img: image to add text to
    :param text: string
    :param pos: (row, col) of bottom left corner of text
    :param font: cv2 param
    :param font_scale: cv2 param
    :param color: cv2 param
    :param bkg_color: 3 or 4 tuple, or None for no bkg
    :param bkg_margin: pixels added to border
    :param thickness: cv2 param, or scaled with font_scale
    :param linestyle: cv2 param
    :param y_axis_flip: flip image coords?
    :param dry_run: return text width and height instead of changing image (img can be None)
    """
    thickness = thickness if thickness is not None else int(np.ceil(font_scale))
    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    if dry_run:
        return w, h

    if bkg_color is not None:
        raise NotImplementedError("text background colors")

    if v_align == 'middle':
        pos = [pos[0], int(pos[1] + h / 2)]
    if v_align == 'top':
        pos = [pos[0], pos[1] + h]

    if h_align == 'right':
        pos = [pos[0] - w, pos[1]]
    elif h_align == 'middle':
        pos = [int(pos[0] - w / 2), pos[1]]

    if y_axis_flip:
        pos = (pos[0], img.shape[0] - pos[1])

    cv2.putText(img, text, pos, font, font_scale, color, thickness,
                linestyle)


def abs_coords_to_bbox_rel(pos, bbox):
    """
    Get the relative coordinates within the bounding box
    :param pos:  x,y coords
    :param bbox: dict with 'top', 'left', 'right','bottom' fields, or list(list(x_min, x_max), list(y_min, y_max))
    :return: x, y in [0.0, 1.0] (if inside, etc.)
    """

    if isinstance(bbox, dict):
        x_rel = float(pos[0] - bbox['left']) / float(bbox['right'] - bbox['left'])
        y_rel = float(pos[1] - bbox['top']) / float(bbox['bottom'] - bbox['top'])

    elif isinstance(bbox, (tuple, list)):
        x_rel = float(pos[0] - bbox[0][0]) / float(bbox[0][1] - bbox[0][0])
        y_rel = float(pos[1] - bbox[1][0]) / float(bbox[1][1] - bbox[1][0])
    else:
        raise Exception("Bounding box must be dict or list/tuple.")
    return x_rel, y_rel


def in_unit_box(rel_coords):
    x_rel, y_rel = rel_coords
    if x_rel < 0 or x_rel > 1.0 or y_rel < 0 or y_rel > 1.0:
        return False
    return True


def make_img_from_color(shape, color, dtype=np.uint8):
    color = np.array(color)
    img = np.tile(color.reshape(color.size, 1, 1),
                  (1, shape[0], shape[1])).astype(dtype)
    img = np.transpose(img, (1, 2, 0))
    img = np.ascontiguousarray(img)
    return img


def draw_box(img, x_span, y_span, color, thickness=1, **kwargs):
    box_points = np.array([[x_span[0], y_span[0]],
                           [x_span[1], y_span[0]],
                           [x_span[1], y_span[1]],
                           [x_span[0], y_span[1]]])
    box_points = np.int32([box_points])

    cv2.polylines(img, box_points, isClosed=True, color=color, thickness=thickness,
                  lineType=cv2.LINE_AA, **kwargs)


def rel_to_abs_span(rel_span, abs_len):
    return np.int64(np.array(rel_span) * abs_len)


def image_overlay(img1, img2, xy_pos):
    """
    Put image 2 on top of image 1. (optional)
    Draw a box around it. (optional)
    to-do, add alpha blending.
    :param img1:  Base image
    :param img2:  image to add
    :param xy_pos:  where is upper left corner of image 2 in image 1
    """
    x_span = xy_pos[0], xy_pos[0] + img2.shape[1]
    y_span = xy_pos[1], xy_pos[1] + img2.shape[0]
    img1[y_span[0]:y_span[1], x_span[0]:x_span[1], ...] = img2


def make_image_from_alpha(alpha, color_low=(0, 0, 0), color_high=(255, 255, 255)):
    color_low = np.array(color_low)
    image_low = make_img_from_color(alpha.shape, color_low, dtype=np.int64)
    image_high = make_img_from_color(alpha.shape, color_high, dtype=np.int64)

    if alpha.dtype == np.uint8:
        alpha = alpha.astype(np.float64) / 255.0
    alpha = alpha[..., None]

    img = (1. - alpha) * image_low + alpha * image_high
    img[img < 0] = 0
    img[img > 255] = 255
    return np.uint8(img)


def eval_ellipse(t, ecc, phi_rad, h):
    _, b = get_ellipse_axes(h, ecc, phi_rad)
    radius = b / np.sqrt(1. - (ecc * np.cos(t + phi_rad)) ** 2.)
    if isinstance(t, np.ndarray):
        return np.stack((radius * np.cos(t),
                         radius * np.sin(t)), axis=1)
    return np.array((radius * np.cos(t),
                     radius * np.sin(t))).reshape(1, 2)


def floats_to_pixels(xy, origin):
    """note, returns x,y not y,x"""
    xy = np.int32(xy)
    xy[:, 1] = -xy[:, 1]
    xy += origin
    return xy


def get_ellipse_axes(h, ecc, phi_rad):
    """
    Calculate the major/minor axes of an ellipse.
    from https://math.stackexchange.com/questions/1535774/height-of-a-rotated-ellipse
    :param h: half distance between staff lines
    :param ecc: eccentricity of notehead
    :param phi_rad: rotation of notehead in radians
    :returns: a, b, major/minor axis lengths
    """
    cos2phi = np.cos(2. * phi_rad)
    qd2 = (1. / (1. - ecc ** 2.)) * (1. - cos2phi) + cos2phi + 1
    b = h / np.sqrt(qd2 / 2.)
    a = np.sqrt(b ** 2. / (1. - ecc ** 2.))
    y = np.sqrt(0.5 * (a ** 2. + b ** 2. - (a ** 2. - b ** 2.) * np.cos(2. * phi_rad)))
    assert np.abs(y - h) < 1e-8, "Error calculating ellipse params."
    return a, b


def _draw_some_ellipses():
    e = []
    h = 100
    t = np.linspace(0, np.pi * 2, 4000)
    for phi in np.linspace(0, 90., 10):
        a, b = get_ellipse_axes(h, 0.9, phi_rad=phi)
        e = np.sqrt(1. - (b ** 2. / a ** 2.))
        r = b / np.sqrt(1. - (e * np.cos(t + phi)) ** 2.)
        x = r * np.cos(t)
        y = r * np.sin(t)
        plt.plot(x, y)
    xlim = plt.xlim()
    plt.plot(xlim, (h, h), 'k-')
    plt.plot(xlim, (-h, -h), 'k-')
    plt.axis('equal')
    plt.show()


def test_text(size):
    imgc = np.zeros((800, 200), dtype=np.uint8)
    imgf = np.zeros((800, 200), dtype=np.uint8)

    scales = [1.0, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3]
    for i in range(11):
        add_text(imgc, "GG", (5, imgc.shape[0] - 5 - i * 67),
                 font=cv2.FONT_HERSHEY_COMPLEX,
                 font_scale=scales[i],
                 thickness=2)
        add_text(imgf, "GG", (5, imgf.shape[0] - 5 - i * 67),
                 font=cv2.FONT_HERSHEY_COMPLEX, linestyle=0,
                 font_scale=scales[i],
                 thickness=int(np.floor(scales[i])))
    plt.subplot(1, 2, 1)
    plt.imshow(imgf)
    plt.subplot(1, 2, 2)
    plt.imshow(imgc)
    plt.show()

    plt.subplot(1, 2, 1)
    plt.imshow(imgc)
    color_light = (245, 235, 229)
    color_dark = (0, 0, 0)
    plt.subplot(1, 2, 2)
    img5 = make_image_from_alpha(imgc, color_low=color_light, color_high=color_dark)
    plt.imshow(img5)

    plt.show()


def make_n_colors(n, scale=(.8, .69, .46)):
    """
    Make a palette of evenly distanced colors.
    :param n:  how many to make?
    :param scale:  [0.0, 1.0] weights for R, G, and B
    :return:  n x 3 array of colors
    """
    color_range = np.linspace(0., np.pi, n + 1)

    colors = np.vstack([[scale[0] * np.abs(np.sin(color_range + np.pi / 4))],
                        [scale[1] * np.abs(np.sin(color_range + np.pi / 2.))],
                        [scale[2] * np.abs(np.sin(color_range))]])

    odds = colors[:, 1::2]
    colors[:,1::2] = odds[::-1]
    colors = colors[:, :-1]
    return colors.T


def show_colors():
    n = 50
    c = make_n_colors(n)
    plt.subplot(1, 2, 1)

    plt.plot(c[:, 0], 'r-')
    plt.plot(c[:, 1], 'go')
    plt.plot(c[:, 2], 'bx')
    plt.subplot(1, 2, 2)
    for i in range(n):
        x = (0, 1, np.nan)
        y = (i, i, np.nan)
        plt.plot(x, y, '-', linewidth=10, color=c[i, :])
    plt.show()


if __name__ == "__main__":
    show_colors()
