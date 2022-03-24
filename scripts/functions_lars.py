import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import binary_opening, binary_closing  # , binary_erosion, binary_dilation
from scipy.signal import fftconvolve
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage.feature import canny
from copy import deepcopy
import cv2


def z_projection_and_outlier_cutoff(data, max_slices, which_channel=0, mode='max'):

    """
    Converts the Odemis data into an image for further use in the automatic ROI detection.
    z-stacks get projected into a single image with a maximum number of slices used.
    Outliers in the image are cut off by looking at the intensity histogram of the image.

    Parameters:
        data (list):            The raw imaging data that needs to be converted, can be a z-stack or single image.
        max_slices (int):       The maximum number of slices used in the z-projection.
                                It is always centered around the slice in the middle of the z-stack.
        which_channel (int):    If there are multiple channels in the data, you can specify which you need.
        mode (string):          To indicate which method to use with the z-projection, two options:
                                    'max': maximum intensity projection
                                    'mean': average intensity projection

    Returns:
        img (ndarray): the image returned as a numpy array
    """

    if data[0].shape[0] == 1:
        num_z = len(data[which_channel][0][0])
        print("#z-slices: {}".format(num_z))
        if num_z <= max_slices:  # if there are too many slices, take the #max_slices in the middle
            img = np.array(data[which_channel][0][0], dtype=int)
        else:
            s = int((num_z - max_slices) / 2)
            img = np.array(data[which_channel][0][0][s:-s], dtype=int)

        histo, bins = np.histogram(img, bins=2000)
        histo = np.cumsum(histo)
        histo = histo / histo[-1]
        plc = np.min(np.where(histo > 0.99))
        thr_out_before = bins[plc] / 2 + bins[plc + 1] / 2
        img[img > thr_out_before] = thr_out_before

        if mode == 'max':
            img = np.max(img, axis=0)
        elif mode == 'mean':
            img = np.mean(img, axis=0)
        else:
            raise Warning("mode input not recognized, maximum intensity projection is used.")
            img = np.max(img, axis=0)

    else:
        print("#z-slices before: 1")
        img = np.array(data[which_channel], dtype=int)
        del data

        histo, bins = np.histogram(img, bins=2000)
        histo = np.cumsum(histo)
        histo = histo / histo[-1]
        plc = np.min(np.where(histo > 0.99))
        thr_out_before = bins[plc] / 2 + bins[plc + 1] / 2
        img[img > thr_out_before] = thr_out_before

    return img


def rescaling(img_before, img_after):

    """
    To rescale a smaller image (k x l) to a bigger image (m x n).
    The dimension of the smaller image goes from (k x l) to (m x n).
    If they are of the same size, nothing happens and the input is returned.

    Parameters:
        img_before (ndarray):   The first image to compare.
        img_after (ndarray):    The second image to compare.

    Returns:
        img_before (ndarray):   If it was the smaller image, the rescaled image, otherwise the same as the input image.
        img_after (ndarray):    If it was the smaller image, the rescaled image, otherwise the same as the input image.
    """

    if img_after.shape == img_before.shape:
        return img_before, img_after

    if img_after.shape > img_before.shape:
        img_rescaled = np.zeros(img_after.shape)
        d_pix = img_before.shape[0] / img_after.shape[0]  # here we assume that the difference in x is the same
        # as in y as pixels are squares
        for i in range(img_before.shape[0]):
            x_vals = np.arange(0, len(img_before[i, :]), 1)
            y_vals = img_before[i, :]
            x_new = np.arange(0, len(img_before[i, :]), d_pix)
            y_new = np.interp(x_new, x_vals, y_vals)
            img_rescaled[i, :] = y_new

        for i in range(img_rescaled.shape[1]):
            y_vals = np.arange(0, len(img_before[:, 0]), 1)
            x_vals = img_rescaled[:img_before.shape[0], i]
            y_new = np.arange(0, len(img_before[:, 0]), d_pix)
            x_new = np.interp(y_new, y_vals, x_vals)
            img_rescaled[:, i] = x_new
        # img_before = img_rescaled
        return img_rescaled, img_after

    # not necessary anymore to have it copied! change it later.
    else:  # copied the code but then for the images swapped, this is better memory wise (I think):
        img_rescaled = np.zeros(img_before.shape)
        d_pix = img_after.shape[0] / img_before.shape[0]  # here we assume that the difference in x is the same
        # as in y as pixels are squares
        for i in range(img_after.shape[0]):
            x_vals = np.arange(0, len(img_after[i, :]), 1)
            y_vals = img_after[i, :]
            x_new = np.arange(0, len(img_after[i, :]), d_pix)
            y_new = np.interp(x_new, x_vals, y_vals)
            img_rescaled[i, :] = y_new

        for i in range(img_rescaled.shape[1]):
            y_vals = np.arange(0, len(img_after[:, 0]), 1)
            x_vals = img_rescaled[:img_after.shape[0], i]
            y_new = np.arange(0, len(img_after[:, 0]), d_pix)
            x_new = np.interp(y_new, y_vals, x_vals)
            img_rescaled[:, i] = x_new
        # img_after = img_rescaled
        return img_before, img_rescaled


def overlay(img_before, img_after, max_shift=5):
    # max shift is between 1 and inf, the higher, the higher the shift can be
    # if it is 1 no constraints will be given on the shift
    mean_after = np.mean(img_after)
    img_before = img_before - np.mean(img_before)
    img_after = img_after - mean_after
    conv = fftconvolve(img_before, img_after[::-1, ::-1], mode='same')

    if (int(conv.shape[0] / max_shift) > 0) & (max_shift > 1):  # constraints
        conv[:int(conv.shape[0] / max_shift), :] = 0
        conv[-int(conv.shape[0] / max_shift):, :] = 0
        conv[:, :int(conv.shape[1] / max_shift)] = 0
        conv[:, -int(conv.shape[1] / max_shift):] = 0

    shift = np.where(conv == np.max(conv))
    shift = np.asarray(shift)
    shift[0] = shift[0] - img_after.shape[0] / 2
    shift[1] = shift[1] - img_after.shape[1] / 2

    img_after = img_after + mean_after

    dy_pix = int(shift[0])
    dx_pix = int(shift[1])

    print("dx is {} pixels".format(dx_pix))
    print("dy is {} pixels".format(dy_pix))

    # shifting the after milling image towards the before milling image to overlap nicely
    if dx_pix > 0:
        img_after[:, dx_pix:] = img_after[:, :-dx_pix]
        img_after[:, :dx_pix] = np.max(img_after)
    elif dx_pix < 0:
        img_after[:, :dx_pix] = img_after[:, -dx_pix:]
        img_after[:, dx_pix:] = np.max(img_after)
    if dy_pix > 0:
        img_after[dy_pix:, :] = img_after[:-dy_pix, :]
        img_after[:dy_pix, :] = np.max(img_after)
    elif dy_pix < 0:
        img_after[:dy_pix, :] = img_after[-dy_pix:, :]
        img_after[dy_pix:, :] = np.max(img_after)

    return img_after


def blur_and_norm(img_before, img_after, blur=5):
    img_before_blurred = gaussian_filter(img_before, sigma=blur)
    img_after_blurred = gaussian_filter(img_after, sigma=blur)

    # preprocessing steps of the images: normalization of blurred images to [0,1] interval
    base_lvl_before = np.min(img_before_blurred)
    base_lvl_after = np.min(img_after_blurred)
    img_before_blurred = (img_before_blurred - base_lvl_before) / (np.max(img_before_blurred) - base_lvl_before)
    img_after_blurred = (img_after_blurred - base_lvl_after) / (np.max(img_after_blurred) - base_lvl_after)

    # preprocessing steps of the images: normalization of initial images to [0,1] interval
    img_before = (img_before - base_lvl_before) / (np.max(img_before) - base_lvl_before)
    img_after = (img_after - base_lvl_after) / (np.max(img_after) - base_lvl_after)

    # you take base_lvl of blurred image because of noise in the initial image (the range would be bigger than needed)
    img_before[img_before < 0] = 0
    img_after[img_after < 0] = 0

    return img_before, img_after, img_before_blurred, img_after_blurred


def create_mask(img_before_blurred, img_after_blurred, threshold_mask, blur, squaring=False):
    # calculating the difference between the two images and creating a mask
    diff = img_before_blurred - 1.5 * img_after_blurred  # times 1.5 to account for intensity differences (more robust)
    mask = diff >= threshold_mask
    mask = binary_opening(mask, iterations=int(blur / 2))  # First opening
    # mask2 = binary_erosion(mask2, iterations=5)  # if the edges give too much signal we can erode the mask a bit more

    # to crop and/or square the image around the ROI if possible and wanted: cropping = True and/or squaring = True
    index_mask = np.where(mask)

    # squaring the mask
    if squaring & (len(index_mask[0]) != 0):
        x_min = np.min(index_mask[1])
        y_min = np.min(index_mask[0])
        x_max = np.max(index_mask[1])
        y_max = np.max(index_mask[0])
        mask[y_min:y_max, x_min:x_max] = True

    return mask


def create_masked_img(img_after, mask, cropping=False):
    # getting the after milling signal only in the mask
    masked_img = img_after * mask
    index_mask = np.where(mask)
    # cropping the image to the mask
    if cropping & (len(index_mask[0]) != 0):
        x_min = np.min(index_mask[1])
        y_min = np.min(index_mask[0])
        x_max = np.max(index_mask[1])
        y_max = np.max(index_mask[0])

        masked_img = masked_img[y_min:y_max, x_min:x_max]

    else:  # for the extent in the plotting
        x_min = 0
        y_min = 0
        x_max = img_after.shape[1]
        y_max = img_after.shape[0]

    masked_img[0, 0] = 1  # to give the images the same color scale
    extents = (x_min, x_max, y_max, y_min)
    return masked_img, extents


def create_binary_end_image(masked_img, threshold, open_close=True):
    binary_end_result = masked_img >= threshold

    # getting rid of too small ROIs and noise after the threshold
    if open_close:
        binary_end_result = binary_closing(binary_opening(binary_closing(binary_end_result, iterations=1),
                                                          iterations=4), iterations=4)
    return binary_end_result


def detect_blobs(binary_end_result2, min_circ=0, max_circ=1, min_area=0, max_area=np.inf, min_in=0, max_in=1,
                 plotting=False):
    im = np.array(binary_end_result2 * 255,
                  dtype=np.uint8)  # cv2.cvtColor(binary_end_result2.astype('uint8'), cv2.COLOR_BGR2GRAY)
    im = cv2.cvtColor(im, cv2.IMREAD_GRAYSCALE)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = False
    params.filterByConvexity = False

    if (min_circ == 0) & (max_circ == 1):
        params.filterByCircularity = False
    else:
        params.filterByCircularity = True
        params.minCircularity = min_circ
        params.maxCircularity = max_circ

    if (min_area == 0) & (max_area == np.inf):
        params.filterByArea = False
    else:
        params.filterByArea = True
        params.minArea = min_area
        params.maxArea = max_area

    if (min_in == 0) & (max_in == 1):
        params.filterByInertia = False
    else:
        params.filterByInertia = True
        params.minInertiaRatio = min_in
        params.maxInertiaRatio = max_in

    ver = cv2.__version__.split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    key_points = detector.detect(im)

    yxr = np.zeros((len(key_points), 3), dtype=int)
    for u in range(len(key_points)):
        yxr[u, 1] = int(key_points[u].pt[0])
        yxr[u, 0] = int(key_points[u].pt[1])
        yxr[u, 2] = int(key_points[u].size / 2)  # the size is the diameter

    # Draw them
    if plotting:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
        for center_y, center_x, radius in zip(yxr[:, 0], yxr[:, 1], yxr[:, 2]):
            circy, circx = circle_perimeter(center_y, center_x, radius,
                                            shape=im.shape)
            im[circy, circx] = (220, 20, 20, 255)

        ax.imshow(im)
        plt.show()

    return key_points, yxr


def plot_end_results(img_before, img_after, img_before_blurred, img_after_blurred, mask, masked_img, masked_img2,
                     binary_end_result1, binary_end_result2, cropping, extents, extents2):
    # plotting the results
    x_min = extents2[0]
    x_max = extents2[1]
    y_min = extents2[3]
    y_max = extents2[2]
    ints_before = np.max(img_before_blurred) * 1.1
    ints_after = np.max(img_after_blurred) * 1.1
    cv2.line(img_before_blurred, (x_min, y_min), (x_max, y_min), ints_before, 5)
    cv2.line(img_before_blurred, (x_max, y_min), (x_max, y_max), ints_before, 5)
    cv2.line(img_before_blurred, (x_min, y_max), (x_max, y_max), ints_before, 5)
    cv2.line(img_before_blurred, (x_min, y_min), (x_min, y_max), ints_before, 5)
    cv2.line(img_after_blurred, (x_min, y_min), (x_max, y_min), ints_after, 5)
    cv2.line(img_after_blurred, (x_max, y_min), (x_max, y_max), ints_after, 5)
    cv2.line(img_after_blurred, (x_min, y_max), (x_max, y_max), ints_after, 5)
    cv2.line(img_after_blurred, (x_min, y_min), (x_min, y_max), ints_after, 5)
    fig, ax = plt.subplots(4, 2)
    ax[0, 0].imshow(img_before)
    ax[0, 0].set_title("before")
    ax[0, 1].imshow(img_after)
    ax[0, 1].set_title("after")
    ax[1, 0].imshow(img_before_blurred)
    ax[1, 1].imshow(img_after_blurred)
    if cropping & (np.sum(mask) > 0):
        ax[2, 0].imshow(masked_img, extent=extents)
        ax[2, 1].imshow(masked_img2, extent=extents2)
        ax[3, 0].imshow(binary_end_result1, extent=extents)
        ax[3, 1].imshow(binary_end_result2, extent=extents2)
    else:
        ax[2, 0].imshow(masked_img)
        ax[2, 1].imshow(masked_img2)
        ax[3, 0].imshow(binary_end_result1)
        ax[3, 1].imshow(binary_end_result2)
    ax[2, 0].set_title("masked without opening")
    ax[2, 1].set_title("masked with opening")

    ax[3, 0].set_title("end without 2nd opening")

    ax[3, 1].set_title("end with 2nd opening")
    plt.show()


def find_x_or_y_pos_milling_site(img_before_blurred, img_after_blurred, ax='x'):
    if ax == 'x':
        projection_before = np.sum(img_before_blurred, axis=0)
        projection_after = np.sum(img_after_blurred, axis=0)
    elif ax == 'y':
        projection_before = np.sum(img_before_blurred, axis=1)
        projection_after = np.sum(img_after_blurred, axis=1)
    else:
        raise NameError("ax does not have the correct input")
    diff_proj = projection_before - projection_after
    mid_milling_site = np.where(diff_proj == np.max(diff_proj))[0][0]
    return mid_milling_site


def find_lines(img_after_blurred, blur=10, low_thres_edges=0.1, high_thres_edges=2.5, angle_res=np.pi/180, 
               line_thres=1, min_lin_lines=1/45, max_line_gap=1/40):
    image = img_as_ubyte(img_after_blurred)
    image = cv2.bitwise_not(image)

    shape_1 = image.shape[1]
    # blur = 10
    # low_thres_edges = 0.1
    # high_thres_edges = 2.5
    # angle_res = np.pi / 180
    # line_thres = 1
    min_len_lines = shape_1 * min_lin_lines
    max_line_gap = shape_1 * max_line_gap

    edges = img_as_ubyte(canny(image, sigma=blur, low_threshold=low_thres_edges, high_threshold=high_thres_edges))
    lines = cv2.HoughLinesP(edges, 1, angle_res, threshold=line_thres, minLineLength=min_len_lines,
                            maxLineGap=max_line_gap)
    if lines is not None:
        print("{} lines detected.".format(len(lines)))
        x_lines = (lines.T[2] / 2 + lines.T[0] / 2).T.reshape((len(lines),))  # the middle points of the line
        y_lines = (lines.T[3] / 2 + lines.T[1] / 2).T.reshape((len(lines),))
        return x_lines, y_lines, lines, edges
    else:
        print("No lines are detected at all!")
        return None, None, None, None


def calculate_angles(lines):
    if lines is None:
        return None
    angle_lines = np.zeros(lines.shape[0])
    for i in range(lines.shape[0]):
        if lines[i][0][2] != lines[i][0][0]:
            angle_lines[i] = np.arctan((lines[i][0][3] - lines[i][0][1]) / (lines[i][0][2] - lines[i][0][0]))
            if angle_lines[i] < 0:
                angle_lines[i] = angle_lines[i] + np.pi
        else:
            angle_lines[i] = np.pi / 2
    return angle_lines


def combine_and_constraint_lines(x_lines, y_lines, lines, angle_lines, mid_milling_site, img_shape, 
                                 x_width_constraint=1/4, y_width_constraint=3/4, angle_constraint=np.pi/7, 
                                 max_dist=1/30, max_diff_angle=np.pi/12):
    if lines is None:
        return None, None, None, None
    x_constraint = img_shape[1] * x_width_constraint / 2  
    # in which region you leave the lines exist : mid_mil_site +- x_constraint
    y_constraint = img_shape[0] * y_width_constraint / 2  
    # in which region you leave the lines exist: y_const < y < y_shape - y_const
    # angle_constraint = np.pi / 7  # exclude lines with angles < angle_constraint and > np.pi-angle_constraint
    max_dist_for_merge = img_shape[0] * max_dist  # the max distance lines can have for merging
    # max_diff_angle = np.pi / 12  # the max angle difference lines can have for merging

    print("mid_milling_site = {}".format(mid_milling_site))
    print("lower x boundary = {}".format(mid_milling_site - x_constraint))
    print("upper x boundary = {}".format(mid_milling_site + x_constraint))

    num_lines_present = np.ones(lines.shape[0])
    for i in range(lines.shape[0]):
        x1 = x_lines[i]
        y1 = y_lines[i]
        if (angle_lines[i] <= angle_constraint) | (angle_lines[i] >= (np.pi - angle_constraint)) | \
                (x1 < (mid_milling_site - x_constraint)) | \
                (x1 > (mid_milling_site + x_constraint)) | \
                (y1 > (img_shape[0]/2 + y_constraint)) | (y1 < (img_shape[0]/2 - y_constraint)):
            num_lines_present[i] = 0
        else:
            for j in range(lines.shape[0]):
                if (i != j) & (num_lines_present[i] >= 1) & (num_lines_present[j] >= 1):
                    dist = np.sqrt((x_lines[i] - x_lines[j]) ** 2 + (y_lines[i] - y_lines[j]) ** 2)
                    diff_angle = np.abs(angle_lines[i] - angle_lines[j])
                    if (dist <= max_dist_for_merge) & (diff_angle <= max_diff_angle):
                        if i < j:
                            lines[i][0][0] = (lines[i][0][0] * num_lines_present[i] + lines[j][0][0] *
                                              num_lines_present[j]) / (num_lines_present[i] + num_lines_present[j])
                            lines[i][0][1] = (lines[i][0][1] * num_lines_present[i] + lines[j][0][1] *
                                              num_lines_present[j]) / (num_lines_present[i] + num_lines_present[j])
                            lines[i][0][2] = (lines[i][0][2] * num_lines_present[i] + lines[j][0][2] *
                                              num_lines_present[j]) / (num_lines_present[i] + num_lines_present[j])
                            lines[i][0][3] = (lines[i][0][3] * num_lines_present[i] + lines[j][0][3] *
                                              num_lines_present[j]) / (num_lines_present[i] + num_lines_present[j])
                            angle_lines[i] = (angle_lines[i] * num_lines_present[i]
                                              + angle_lines[j] * num_lines_present[j]) / \
                                             (num_lines_present[i] + num_lines_present[j])
                            x_lines[i] = (x_lines[i] * num_lines_present[i] + x_lines[j] * num_lines_present[
                                j]) / (num_lines_present[i] + num_lines_present[j])
                            y_lines[i] = (y_lines[i] * num_lines_present[i] + y_lines[j] * num_lines_present[
                                j]) / (num_lines_present[i] + num_lines_present[j])
                            num_lines_present[i] = num_lines_present[i] + num_lines_present[j]
                            num_lines_present[j] = 0
                        else:
                            lines[j][0][0] = (lines[i][0][0] * num_lines_present[i] + lines[j][0][0] *
                                              num_lines_present[j]) / (num_lines_present[i] + num_lines_present[j])
                            lines[j][0][1] = (lines[i][0][1] * num_lines_present[i] + lines[j][0][1] *
                                              num_lines_present[j]) / (num_lines_present[i] + num_lines_present[j])
                            lines[j][0][2] = (lines[i][0][2] * num_lines_present[i] + lines[j][0][2] *
                                              num_lines_present[j]) / (num_lines_present[i] + num_lines_present[j])
                            lines[j][0][3] = (lines[i][0][3] * num_lines_present[i] + lines[j][0][3] *
                                              num_lines_present[j]) / (num_lines_present[i] + num_lines_present[j])
                            angle_lines[j] = (angle_lines[i] * num_lines_present[i]
                                              + angle_lines[j] * num_lines_present[j]) / \
                                             (num_lines_present[i] + num_lines_present[j])
                            x_lines[j] = (x_lines[i] * num_lines_present[i] + x_lines[j] * num_lines_present[
                                j]) / (num_lines_present[i] + num_lines_present[j])
                            y_lines[j] = (y_lines[i] * num_lines_present[i] + y_lines[j] * num_lines_present[
                                j]) / (num_lines_present[i] + num_lines_present[j])
                            num_lines_present[j] = num_lines_present[i] + num_lines_present[j]
                            num_lines_present[i] = 0

    lines = lines[num_lines_present > 0]
    x_lines = x_lines[num_lines_present > 0]
    y_lines = y_lines[num_lines_present > 0]
    angle_lines = angle_lines[num_lines_present > 0]

    return x_lines, y_lines, lines, angle_lines
    

def group_single_lines(x_lines, y_lines, lines, angle_lines, max_distance, max_angle_diff=np.pi/16):
    if lines is None:
        return None
    groups = []
    for i in range(len(lines)):
        # print("{}% done".format(i / len(lines) * 100))
        set_in_group = False
        x1 = x_lines[i]
        y1 = y_lines[i]
        angle = angle_lines[i]
        for j in range(len(groups)):
            angle_mean = np.mean(angle_lines[groups[j]])
            x2 = np.mean(x_lines[groups[j]])
            y2 = np.mean(y_lines[groups[j]])
            if np.abs(angle - angle_mean) <= max_angle_diff:
                x3 = (y2 - y1 + np.tan(angle_mean + np.pi / 2 + 1e-10) * x1 - np.tan(angle_mean + 1e-10) * x2) \
                     / (np.tan(angle_mean + np.pi / 2 + 1e-10) - np.tan(angle_mean + 1e-10))
                y3 = np.tan(angle_mean + 1e-10) * (x3 - x2) + y2
                dist = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
                if dist <= max_distance:  # img_after.shape[1] / 20
                    set_in_group = True
                    groups[j].append(i)
        if not set_in_group:
            groups.append([i])

    for i in range(len(groups)):
        groups[i] = np.unique(groups[i])
    return groups


def couple_groups_of_lines(groups, x_lines, y_lines, angle_lines, min_dist, max_dist, max_angle_diff=np.pi/8):
    if groups is None:
        return None
    size_groups_combined = np.zeros((len(groups), len(groups)))
    for i in range(len(groups)):
        for j in np.arange(i + 1, len(groups), 1):
            if np.abs(np.mean(angle_lines[groups[i]]) - np.mean(angle_lines[groups[j]])) <= max_angle_diff:
                angle_mean = (np.mean(angle_lines[groups[i]]) * len(groups[i])
                              + np.mean(angle_lines[groups[j]]) * len(groups[j])) / \
                             (len(groups[i]) + len(groups[j]))
                x1 = np.mean(x_lines[groups[i]])
                y1 = np.mean(y_lines[groups[i]])
                x2 = np.mean(x_lines[groups[j]])
                y2 = np.mean(y_lines[groups[j]])
                x3 = (y2 - y1 + np.tan(angle_mean + np.pi / 2 + 1e-10) * x1 - np.tan(angle_mean + 1e-10) * x2) / \
                     (np.tan(angle_mean + np.pi / 2 + 1e-10) - np.tan(angle_mean + 1e-10))
                y3 = np.tan(angle_mean + 1e-10) * (x3 - x2) + y2
                dist = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
                if (dist < max_dist) & (dist > min_dist):
                    size_groups_combined[i, j] = (len(groups[i]) + len(groups[j]))

    if np.max(size_groups_combined) > 0:
        biggest = np.where(size_groups_combined == np.max(size_groups_combined))  
        # which one consist of the most pieces of small lines
        after_grouping = []
        if len(biggest[0]) > 1:
            merge_big_groups = np.zeros((len(biggest[0]), len(biggest[0])), dtype=bool)

            for k in range(len(biggest[0])):
                for m in np.arange(k + 1, len(biggest[0]), 1):
                    angle_k = (np.mean(angle_lines[groups[biggest[0][k]]]) * len(groups[biggest[0][k]])
                               + np.mean(angle_lines[groups[biggest[1][k]]]) * len(groups[biggest[1][k]])) / \
                              (len(groups[biggest[0][k]]) + len(groups[biggest[1][k]]))
                    angle_l = (np.mean(angle_lines[groups[biggest[0][m]]]) * len(groups[biggest[0][m]])
                               + np.mean(angle_lines[groups[biggest[1][m]]]) * len(groups[biggest[1][m]])) / \
                              (len(groups[biggest[0][m]]) + len(groups[biggest[1][m]]))
                    if np.abs(angle_k - angle_l) <= max_angle_diff:
                        merge_big_groups[k, m] = True
            print("total groups used: {}".format(np.sum(merge_big_groups)))
            if np.sum(merge_big_groups) != 0:
                groups2 = []
                for x, y in np.array(np.where(merge_big_groups)).T:
                    set_in_group2 = False
                    for r in range(len(groups2)):
                        if x in groups2[r] and y in groups2[r]:
                            set_in_group2 = True
                        elif x in groups2[r] and y not in groups2[r]:
                            groups2[r].append(y)
                            set_in_group2 = True
                        elif y in groups2[r] and x not in groups2[r]:
                            groups2[r].append(x)
                            set_in_group2 = True
                    if not set_in_group2:
                        groups2.append([x, y])
                b = 0
                final_group = 0
                for r in range(len(groups2)):
                    groups2[r] = np.unique(groups2[r])
                    if len(groups2[r]) > b:
                        b = len(groups2[r])
                        final_group = r
                for i in groups2[final_group]:
                    for j in groups[biggest[0][i]]:
                        after_grouping.append(j)
                    for k in groups[biggest[1][i]]:
                        after_grouping.append(k)

            else:
                # if no groups merge and there are multiple groups the biggest,
                # just take the first one and hope for the best
                for i in groups[biggest[0][0]]:
                    after_grouping.append(i)
                for j in groups[biggest[1][0]]:
                    after_grouping.append(j)

        else:  # there is only one group
            for i in groups[biggest[0][0]]:
                after_grouping.append(i)
            for j in groups[biggest[1][0]]:
                after_grouping.append(j)
        after_grouping = np.unique(after_grouping)
    else:  # there are no coupled line groups
        after_grouping = np.array([])
    return after_grouping


def show_line_detection_steps(img_after, img_after_blurred, edges, lines, lines2, after_grouping):
    if lines is not None:
        image = img_as_ubyte(img_after_blurred)
        image = cv2.bitwise_not(image)
        image = image * 0.8
        image1 = deepcopy(image)
        image2 = deepcopy(image)
        image3 = deepcopy(image)

        if len(after_grouping) > 0:
            print("{} lines after grouping".format(len(after_grouping)))

            for points in lines2[after_grouping]:
                # Extracted points nested in the list
                x1, y1, x2, y2 = points[0]
                # Draw the lines joining the points on the original image
                cv2.line(image3, (x1, y1), (x2, y2), 255, 2)
        else:
            print("0 lines after grouping")

        for points in lines:
            # Extracted points nested in the list
            x1, y1, x2, y2 = points[0]
            # Draw the lines joining the points on the original image
            cv2.line(image, (x1, y1), (x2, y2), 255, 2)

        for points in lines2:
            # Extracted points nested in the list
            x1, y1, x2, y2 = points[0]
            # Draw the lines joining the points on the original image
            cv2.line(image1, (x1, y1), (x2, y2), 255, 2)

        fig, ax = plt.subplots(ncols=2, nrows=3)
        ax[0, 0].imshow(img_after)
        ax[0, 0].set_title('img_after')
        ax[0, 1].imshow(edges)
        ax[0, 1].set_title('the edges')
        ax[1, 1].imshow(image1)
        ax[1, 1].set_title('similar lines combined')
        ax[1, 0].imshow(image)
        ax[1, 0].set_title('all the lines')
        ax[2, 0].imshow(image2)
        ax[2, 0].set_title('after selection')
        ax[2, 1].imshow(image3)
        ax[2, 1].set_title('after grouping')
        plt.show()
    else:
        print("No lines were detected so nothing can be shown.")
