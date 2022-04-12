import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import binary_opening, binary_closing, binary_erosion, binary_dilation
from scipy.signal import fftconvolve
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage.feature import canny
from skimage.morphology import disk
from skimage.filters.rank import gradient
from copy import deepcopy
from odemis.dataio import tiff
import cv2


def find_focus_z_slice_and_outlier_cutoff(data, milling_pos_y, milling_pos_x, which_channel=0, blur=20,
                                          square_width=1 / 5, num_slices=5, disk_size=5, outlier_cutoff=99, mode='max'):
    if data[0].shape[0] == 1:
        num_z = len(data[which_channel][0][0])
        print("#z-slices: {}".format(num_z))
        dat = data[which_channel][0][0]
        sharp = np.zeros(dat.shape[0])
        selection_element = disk(disk_size)
        histo, bins = np.histogram(dat, bins=2000)
        histo = np.cumsum(histo)
        histo = histo / histo[-1] * 100
        plc = np.min(np.where(histo > outlier_cutoff))
        thr_out = bins[plc] / 2 + bins[plc + 1] / 2
        print("outlier threshold is at {}".format(thr_out))
        dat[dat > thr_out] = thr_out

        minmax = np.zeros(4, dtype=int)  # y_min, y_max, x_min, x_max
        minmax[0] = int(milling_pos_y - dat[0].shape[0] * square_width / 2)
        minmax[1] = int(milling_pos_y + dat[0].shape[0] * square_width / 2)
        minmax[2] = int(milling_pos_x - dat[0].shape[1] * square_width / 2)
        minmax[3] = int(milling_pos_x + dat[0].shape[1] * square_width / 2)

        for w in range(2):
            # x values
            if minmax[w] < 0:
                minmax[w] = 0
            elif minmax[w] > dat[0].shape[0]:
                minmax[w] = dat[0].shape[0]

            if minmax[w+2] < 0:
                minmax[w+2] = 0
            elif minmax[w+2] > dat[0].shape[1]:
                minmax[w+2] = dat[0].shape[1]

        for i in range(dat.shape[0]):
            # print(i)
            # dat[i][dat[i] > thr_out] = thr_out
            img = dat[i][minmax[0]:minmax[1], minmax[2]:minmax[3]]  # [1000:1600, 1000:1600]
            # img[img > thr_out] = thr_out
            img = gaussian_filter(img, sigma=blur)  # 20
            img = np.array(img / np.max(img) * 255, dtype=np.uint8)
            sharp[i] = np.mean(gradient(img, selection_element))
            # if i >= num_slices:
            #     if 1.01*sharp[i] < np.mean(sharp[i-num_slices:i]):
            #         break
        in_focus = np.where(sharp == np.max(sharp))[0][0]

        minmax = np.zeros(2, dtype=int)  # y_min, y_max, x_min, x_max
        minmax[0] = in_focus - int((num_slices-1)/2)
        minmax[1] = in_focus + int((num_slices-1)/2) + 1

        if minmax[0] < 0:
            minmax[0] = 0
        if minmax[1] > dat.shape[0]:
            minmax[1] = dat.shape[0]

        print(f"In focus slice is: {in_focus}.")
        if mode == 'max':
            img = np.max(dat[minmax[0]:minmax[1]], axis=0)
            print(dat[minmax[0]:minmax[1]].shape)
        elif mode == 'mean':
            img = np.mean(dat[minmax[0]:minmax[1]], axis=0)
        else:
            logging.warning("Mode input not recognized, maximum intensity projection is used.")
            img = np.max(dat[minmax[0]:minmax[1]], axis=0)
    else:
        print("#z-slices: 1")
        img = np.array(data[which_channel], dtype=int)
        del data

        histo, bins = np.histogram(img, bins=2000)
        histo = np.cumsum(histo)
        histo = histo / histo[-1] * 100
        plc = np.min(np.where(histo > outlier_cutoff))
        thr_out = bins[plc] / 2 + bins[plc + 1] / 2
        # ax.plot(bins[1:] - (bins[2] - bins[1]) / 2, histo)
        # ax.set_title("Normalized cumulative histogram of #{}".format(e))
        # ax.set_ylabel("#pixels (%)")
        # ax.set_xlabel("Intensity value ()")
        # print("outlier threshold is at {}".format(thr_out))
        # print("max outlier value is {}".format(bins[-1]))
        img[img > thr_out] = thr_out

    return np.array(img, dtype=int)


def z_projection_and_outlier_cutoff(data, max_slices, which_channel=0, outlier_cutoff=99, mode='max'):
    """
    Converts the Odemis data into an image for further use in the automatic ROI detection.
    z-stacks get projected into a single image with a maximum number of slices used.
    Outliers in the image are cut off by looking at the intensity histogram of the image.

    Parameters:
        data (list):            The raw imaging data that needs to be converted, can be a z-stack or single image.
        max_slices (int):       The maximum number of slices used in the z-projection.
                                It is always centered around the slice in the middle of the z-stack.
        which_channel (int):    If there are multiple channels in the data, you can specify which you need.
        outlier_cutoff (float): At which percentage the intensity histogram should be cut off.
        mode (string):          To indicate which method to use with the z-projection, two options:
                                    'max': maximum intensity projection
                                    'mean': average intensity projection

    Returns:
        img (ndarray): the image returned as a numpy array
    """

    # fig, ax = plt.subplots()
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
        histo = histo / histo[-1] * 100
        plc = np.min(np.where(histo > outlier_cutoff))
        thr_out = bins[plc] / 2 + bins[plc + 1] / 2
        # ax.plot(bins[1:]-(bins[2]-bins[1])/2, histo)
        # ax.set_title("Normalized cumulative histogram of #{}".format(e))
        # ax.set_ylabel("#pixels (%)")
        # ax.set_xlabel("Intensity value ()")
        print("outlier threshold is at {}".format(thr_out))
        # print("max outlier value is {}".format(bins[-1]))
        img[img > thr_out] = thr_out

        if mode == 'max':
            img = np.max(img, axis=0)
        elif mode == 'mean':
            img = np.mean(img, axis=0)
        else:
            logging.warning("Mode input not recognized, maximum intensity projection is used.")
            img = np.max(img, axis=0)

    else:
        print("#z-slices before: 1")
        img = np.array(data[which_channel], dtype=int)
        del data

        histo, bins = np.histogram(img, bins=2000)
        histo = np.cumsum(histo)
        histo = histo / histo[-1] * 100
        plc = np.min(np.where(histo > outlier_cutoff))
        thr_out = bins[plc] / 2 + bins[plc + 1] / 2
        # ax.plot(bins[1:] - (bins[2] - bins[1]) / 2, histo)
        # ax.set_title("Normalized cumulative histogram of #{}".format(e))
        # ax.set_ylabel("#pixels (%)")
        # ax.set_xlabel("Intensity value ()")
        # print("outlier threshold is at {}".format(thr_out))
        # print("max outlier value is {}".format(bins[-1]))
        img[img > thr_out] = thr_out

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

    if img_before.shape == img_after.shape:
        return img_before, img_after, 1

    if img_after.shape > img_before.shape:
        print("image shapes are not equal.")
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
        return img_rescaled, img_after, img_before.shape[0]/img_after.shape[0]

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
        return img_before, img_rescaled, img_before.shape[0]/img_after.shape[0]


def overlay(img_before, img_after, max_shift=5):
    """
    Shift the img_after in x and y position to have the best overlay with img_before.
    This is done with finding the maximum in a convolution.

    Parameters:
        img_before (ndarray):   The image which will be shifted to for the best overlay.
        img_after (ndarray):    The image which will shift to img_before for the best overlay.
        max_shift (int):        It is assumed that the images are roughly at the same position,
                                so no large shifts are needed. On all sides of the convolution image,
                                1/max_shift of the image is set to be zero. So with 1/8 you have a max shift of 3/8 in
                                x and y direction (-3/8*img_shape, 3/8*img_shape). If you don't want a constraint on
                                the shift, set max_shift to <=1

    Returns:
        img_after (ndarray):    The shifted image, the values outside the boundaries of the image are set to the max
                                value of the image so that in further processing steps those regions are not used.
    """

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

    return img_after, shift


def get_image(data_paths_before, data_paths_after, channel_before, channel_after, max_slices=30, mode='in_focus',
              proj_mode='max', blur=25):
    if mode == 'in_focus':
        data_before_milling = tiff.read_data(data_paths_before)
        meta_before = data_before_milling[channel_before].metadata
        data_after_milling = tiff.read_data(data_paths_after)
        meta_after = data_after_milling[channel_after].metadata
        if data_before_milling[0].shape[0] != 1 and data_after_milling[0].shape[0] != 1:
            img_before = z_projection_and_outlier_cutoff(data_before_milling, max_slices, channel_before,
                                                         mode=proj_mode)
            img_after = z_projection_and_outlier_cutoff(data_after_milling, max_slices, channel_after, mode=proj_mode)
            return img_before, img_after, meta_before, meta_after

        img_before = z_projection_and_outlier_cutoff(data_before_milling, max_slices, channel_before,
                                                     mode=proj_mode)
        img_after = z_projection_and_outlier_cutoff(data_after_milling, max_slices, channel_after, mode=proj_mode)

        # rescaling one image to the other if necessary
        img_before, img_after, magni = rescaling(img_before, img_after)
        print(f"scaling factor is: {magni}")
        # calculate the shift between the two images
        img_after, shift = overlay(img_before, img_after, max_shift=4)  # max_shift between 1 and inf
        # shift is [dy, dx]

        # preprocessing steps of the images: blurring the image
        img_before, img_after, img_before_blurred, img_after_blurred = blur_and_norm(img_before, img_after, blur=blur)

        milling_x_pos = find_x_or_y_pos_milling_site(img_before_blurred, img_after_blurred, ax='x')
        milling_y_pos = find_x_or_y_pos_milling_site(img_before_blurred, img_after_blurred, ax='y')

        if magni == 1:
            img_before = find_focus_z_slice_and_outlier_cutoff(data_before_milling, milling_y_pos, milling_x_pos,
                                                               channel_before, blur)
            img_after = find_focus_z_slice_and_outlier_cutoff(data_after_milling, milling_y_pos - shift[0],
                                                              milling_x_pos - shift[1], channel_after, blur)
        elif magni > 1:
            img_before = find_focus_z_slice_and_outlier_cutoff(data_before_milling, milling_y_pos, milling_x_pos,
                                                               channel_before, blur)
            img_after = find_focus_z_slice_and_outlier_cutoff(data_after_milling, (milling_y_pos - shift[0]) / magni,
                                                              (milling_x_pos - shift[1]) / magni, channel_after, blur)
        elif magni < 1:
            img_before = find_focus_z_slice_and_outlier_cutoff(data_before_milling, (milling_y_pos - shift[0]) * magni,
                                                               (milling_x_pos - shift[1]) * magni, channel_before, blur)
            img_after = find_focus_z_slice_and_outlier_cutoff(data_after_milling, milling_y_pos, milling_x_pos,
                                                              channel_after, blur)
    elif mode == 'projection':
        data_before_milling = tiff.read_data(data_paths_before)
        meta_before = data_before_milling[channel_before].metadata
        data_after_milling = tiff.read_data(data_paths_after)
        meta_after = data_after_milling[channel_after].metadata

        img_before = z_projection_and_outlier_cutoff(data_before_milling, max_slices, channel_before,
                                                     mode=proj_mode)
        img_after = z_projection_and_outlier_cutoff(data_after_milling, max_slices, channel_after, mode=proj_mode)

    else:
        raise NameError("Input for mode not recognized. It can be 'projection' or 'in_focus'.")
    return img_before, img_after, meta_before, meta_after


def blur_and_norm(img_before, img_after, blur=25):
    """
    Because the images before and after milling have different intensity profiles and do not have the exact same noise,
    normalization and blurring is required for being able to compare the two images. Here the two images are first
    blurred with a Gaussian filter and afterwards normalized to a range from 0 to 1. The unblurred images are also
    normalized with their own maximum and the minimum of the blurred images. The minimum of the blurred images are used
    because everything below this is probably noise.

    Parameters:
        img_before (ndarray):   The image before milling to be blurred and normalized.
        img_after (ndarray):    The image after milling to be blurred and normalized.
        blur (int):             The sigma for the gaussian blurring, the higher the number, the more blurring occurs.

    Returns:
        img_before (ndarray):           The unblurred, normalized image before milling.
        img_after (ndarray):            The unblurred, normalized image after milling.
        img_before_blurred (ndarray):   The blurred, normalized image before milling.
        img_after_blurred (ndarray):    The blurred, normalized image after milling.
    """

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


def create_diff_mask(img_before_blurred, img_after_blurred, threshold_mask=0.3, open_iter=12, ero_iter=0,
                     squaring=False):
    """
    Here a mask is created with looking at the difference in intensities between the blurred images before and after
    milling. A threshold is set for the difference and a binary opening is performed to get rid of too small
    masks (optional but recommended). A binary erosion and a squaring of the mask also can be done.

    Parameters:
        img_before_blurred (ndarray):   The blurred image before milling.
        img_after_blurred (ndarray):    The blurred image after milling.
        threshold_mask (float):         The threshold at which the difference in intensities needs to be to pass.
        open_iter (int):                How many iterations of binary opening you want to perform.
        ero_iter (int):                 How many iterations of binary erosion you want to perform.
        squaring (bool):                If set to true, a square is taken around the mask becomes the mask itself.

    Returns:
        mask (ndarray):                 The mask created in the form of a boolean numpy array.
    """

    # calculating the difference between the two images and creating a mask
    diff = img_before_blurred - 1.5 * img_after_blurred  # times 1.5 to account for intensity differences (more robust)
    mask = diff >= threshold_mask
    if open_iter >= 1:
        mask = binary_opening(mask, iterations=open_iter)  # First opening int(blur / 2)
    if ero_iter >= 1:
        # if the edges give too much signal we can erode the mask a bit more
        mask = binary_erosion(mask, iterations=ero_iter)

    # to crop and/or square the image around the ROI if possible and wanted: cropping = True and/or squaring = True
    index_mask = np.where(mask)

    # squaring the mask
    if squaring & (len(index_mask[0]) != 0):
        x_min = np.min(index_mask[1])
        y_min = np.min(index_mask[0])
        x_max = np.max(index_mask[1])
        y_max = np.max(index_mask[0])
        # if x_max - x_min < mask.shape[1]/3:
        mask[y_min:y_max, x_min:x_max] = True

    return mask


def find_x_or_y_pos_milling_site(img_before_blurred, img_after_blurred, ax='x'):
    """
    Find the rough position of the milling site in x or y direction. This is done by projection the images on one of
    the axes and look for a maximum in the difference.

    Parameters:
        img_before_blurred (ndarray):   The image before milling.
        img_after_blurred (ndarray):    The image after milling.
        ax (string):                    A string which says which axes to project on (x or y)

    Returns:
        mid_milling_site (int):         The x or y position of the milling site in pixels.
    """

    if ax == 'x':
        projection_before = np.sum(img_before_blurred, axis=0)
        projection_after = np.sum(img_after_blurred, axis=0)
    elif ax == 'y':
        projection_before = np.sum(img_before_blurred, axis=1)
        minimal = np.min(projection_before)
        projection_before[:int(img_before_blurred.shape[0]/6)] = minimal
        projection_before[-int(img_before_blurred.shape[0] / 6):] = minimal

        projection_after = np.sum(img_after_blurred, axis=1)
        # minimal = np.min(projection_after)
        # projection_after[:int(img_after_blurred.shape[0] / 6), :] = minimal
        # projection_after[-int(img_after_blurred.shape[0] / 6):, :] = minimal
    else:
        raise NameError("ax does not have the correct input")
    diff_proj = projection_before - projection_after
    mid_milling_site = np.where(diff_proj == np.max(diff_proj))[0][0]
    return mid_milling_site


def find_lines(img_after_blurred, blur=10, low_thres_edges=0.1, high_thres_edges=2.5, angle_res=np.pi / 180,
               line_thres=1, min_len_lines=1 / 45, max_line_gap=1 / 40):
    """
    This function finds all the lines in an image with the canny (edge detection) function of skimage and the line
    detection function of opencv. You can set constraints to this line detection with the parameters of this function.
    If no lines are found, None is returned for everything.

    For more information on the canny and HoughLines functions:
    https://scikit-image.org/docs/stable/api/skimage.feature.html?highlight=module%20feature#skimage.feature.canny
    https://docs.opencv.org/3.4/d3/de6/tutorial_js_houghlines.html

    Parameters:
        img_after_blurred (ndarray):    The image on which line detection is used.
        blur (int):                     The (extra) blurring done on the image for edge detection.
        low_thres_edges (float):        The lower hysteresis threshold passed into the canny edge detection.
        high_thres_edges (float):       The upper hysteresis threshold passed into the canny edge detection.
        angle_res (float):              The angular resolution with which lines are detected in the opencv function.
        line_thres (int):               The quality threshold for lines to be detected.
        min_len_lines (float):          The minimal length of the lines detected as a fraction of the image size.
        max_line_gap (float):           The maximal gap within the detected lines as a fraction of the image size.

    Returns:
        x_lines ():                     The x-center of the lines detected.
        y_lines ():                     The y-center of the lines detected.
        lines (ndarray):                The lines detected in the format (#lines, 1, 4).
                                        The last index gives the positions of the ends of the lines:
                                        (0: x of end 1, 1: y of end 1, 2: x of end 2, 3: y of end 2).
        edges (ndarray):                The output image of the edge detection.
    """

    image = img_as_ubyte(img_after_blurred)
    image = cv2.bitwise_not(image)

    shape_1 = image.shape[1]
    # blur = 10
    # low_thres_edges = 0.1
    # high_thres_edges = 2.5
    # angle_res = np.pi / 180
    # line_thres = 1
    min_len_lines = shape_1 * min_len_lines
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
    """
    Here the angles of the lines are calculated with a simple arctan. It will have the range of 0 to pi.

    Parameters:
        lines (ndarray):        The end positions of the lines as outputted from find_lines.

    Returns:
        angle_lines (ndarray):  The angles of the lines in a single 1D-array.
    """

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
                                 x_width_constraint=1 / 4, y_width_constraint=3 / 4, angle_constraint=np.pi / 7,
                                 max_dist=1 / 25, max_diff_angle=np.pi / 10):
    """
    Here lines which are roughly at the same spot and angle are combined in one line.
    Next to that, also some constraints can be given for which lines to pick out for further processing.
    Lines at an angle of around 0 or pi are discarded because no solution on the boundary here has been implemented.

    Parameters:
        x_lines (ndarray):              The x-centers of the lines.
        y_lines (ndarray):              The y-centers of the lines.
        lines (ndarray):                The end positions of the lines as outputted from find_lines.
        angle_lines (ndarray):          The angles of the lines as outputted from calculate_angles.
        mid_milling_site (int):         The x center position of the milling site.
        img_shape (tuple):              The shape of the image.
        x_width_constraint (float):     The range of the x centers lines should have around the milling site for further
                                        processing as a fraction of the image shape.
        y_width_constraint (float):     The range of the y centers lines should have around the middle of the image
                                        for further processing as a fraction of the image shape.
        angle_constraint (float):       The angle of the lines have to be bigger than the angle_constraint and smaller
                                        than pi - angle_constraint.
        max_dist (float):               The maximum distance lines can have between their center positions to merge.
        max_diff_angle (float):         The maximum difference in angles lines can have to merge.

    Returns:
        x_lines (ndarray):              The x-center of the lines after merging and selection.
        y_lines (ndarray):              The y-center of the lines after merging and selection.
        lines (ndarray):                The end positions of the lines after merging and selection.
        angle_lines (ndarray):          The angles of the lines after merging and selection.
    """

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
                (y1 > (img_shape[0] / 2 + y_constraint)) | (y1 < (img_shape[0] / 2 - y_constraint)):
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


def group_single_lines(x_lines, y_lines, lines, angle_lines, max_distance, max_angle_diff=np.pi / 12):
    """
    Here individual lines are grouped together based on their angle and perpendicular distance between each other.
    The lines here are regarded as lines with infinite length so that always the perpendicular distance between lines
    can be calculated.

    Parameters:
        x_lines (ndarray):              The x-centers of the lines.
        y_lines (ndarray):              The y-centers of the lines.
        lines (ndarray):                The end positions of the lines.
        angle_lines (ndarray):          The angles of the lines.
        max_distance (float):           The maximum perpendicular distance between lines in pixels to group them.
        max_angle_diff (float):         The maximum difference in angles between lines to group them.

    Returns:
        groups (list):                  A list with all the groups created. Each group has the index values of the
                                        lines in that group. So lines[groups[0]] gives you the lines of the first group.

    """

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


def couple_groups_of_lines(groups, x_lines, y_lines, angle_lines, x_pos_mil, min_dist, max_dist,
                           max_angle_diff=np.pi / 8):
    """
    Here the function will look if groups of lines can together form a band which has roughly the same width as the
    milling site. If multiple groups of lines can form bands, the one with the most individual lines will be outputted.
    If these bands have the same number of lines in them, the function looks if they can be merged or not, if not,
    the band closest to the estimated milling site will be used.

    Parameters:
        groups (list):              The groups of individual lines outputted from group_single_lines.
        x_lines (ndarray):          The x-centers of the lines.
        y_lines (ndarray):          The y-centers of the lines.
        angle_lines (ndarray):      The angles of the lines.
        x_pos_mil (int):            The x position of the milling site found by find_x_or_y_pos_milling_site().
        min_dist (float):           The minimum perpendicular distance between groups of lines in pixels to group them.
        max_dist (float):           The maximum perpendicular distance between groups of lines in pixels to group them.
        max_angle_diff (float):     The maximum difference in angles between groups of lines to group them.

    Returns:
        after_grouping (ndarray):   The index values of the lines of the biggest group after the grouping of groups.
                                    These lines (probably) represent the edges of the milling site.

    """

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
                    angle_m = (np.mean(angle_lines[groups[biggest[0][m]]]) * len(groups[biggest[0][m]])
                               + np.mean(angle_lines[groups[biggest[1][m]]]) * len(groups[biggest[1][m]])) / \
                              (len(groups[biggest[0][m]]) + len(groups[biggest[1][m]]))
                    x_k = (np.mean(x_lines[groups[biggest[0][k]]]) * len(groups[biggest[0][k]])
                           + np.mean(x_lines[groups[biggest[1][k]]]) * len(groups[biggest[1][k]])) / \
                          (len(groups[biggest[0][k]]) + len(groups[biggest[1][k]]))
                    y_k = (np.mean(y_lines[groups[biggest[0][k]]]) * len(groups[biggest[0][k]])
                           + np.mean(y_lines[groups[biggest[1][k]]]) * len(groups[biggest[1][k]])) / \
                          (len(groups[biggest[0][k]]) + len(groups[biggest[1][k]]))
                    x_m = (np.mean(x_lines[groups[biggest[0][m]]]) * len(groups[biggest[0][m]])
                           + np.mean(x_lines[groups[biggest[1][m]]]) * len(groups[biggest[1][m]])) / \
                          (len(groups[biggest[0][m]]) + len(groups[biggest[1][m]]))
                    y_m = (np.mean(y_lines[groups[biggest[0][m]]]) * len(groups[biggest[0][m]])
                           + np.mean(y_lines[groups[biggest[1][m]]]) * len(groups[biggest[1][m]])) / \
                          (len(groups[biggest[0][m]]) + len(groups[biggest[1][m]]))
                    distance = np.sqrt((x_k - x_m) ** 2 + (y_k - y_m) ** 2)

                    if (np.abs(angle_k - angle_m) <= max_angle_diff) & (distance <= min_dist):
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
                # just take the first one and hope for the best, THIS DOESN'T WORK
                # if no groups merge, chose the one closest to the x_pos of the milling site.
                r = 0
                d = np.inf
                for i in range(len(biggest[0])):
                    x_mean = (np.mean(x_lines[groups[biggest[0][i]]]) + np.mean(x_lines[groups[biggest[1][i]]])) / 2
                    if np.abs(x_mean - x_pos_mil) < d:
                        r = i
                        d = np.abs(x_mean - x_pos_mil)

                for i in groups[biggest[0][r]]:
                    after_grouping.append(i)
                for j in groups[biggest[1][r]]:
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
    """
    Here the line detection steps are shown in images for visualization of the process.

    Parameters:
        img_after (ndarray):            The image after milling on which the line detection is executed.
        img_after_blurred (ndarray):    The image after milling blurred and normalized.
        edges (ndarray):                The edges image from the canny function in find_lines().
        lines (ndarray):                The lines before combine_and_constraint_lines().
        lines2 (ndarray):               The lines after combine_and_constraint_lines().
        after_grouping (ndarray):       The end result of the lines which (probably) represent the milling site.

    Returns:
        Nothing, only shows the images with matplotlib.pyplot.
    """

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


def create_line_mask(after_grouping, x_lines2, y_lines2, lines2, angle_lines2, img_shape, inv_max_dist=20,
                     max_angle=np.pi / 8, all_groups=False):
    """
    Here a mask is created with the lines found in the line detection. The lines after grouping, are split into two
    groups (left and right) and an average position and angle of both groups is used for creating the boundaries
    of the mask.

    Parameters:
        after_grouping (ndarray):   The indexes of the lines outputted of the line detection.
        x_lines2 (ndarray):         The center x-positions of the lines.
        y_lines2 (ndarray):         The center y-positions of the lines.
        lines2 (ndarray):           The end points of the lines.
        angle_lines2 (ndarray):     The angles of the lines.
        img_shape (tuple):          The shape of the image after milling.
        inv_max_dist (int):         For grouping the lines in left and right, the lines within a group cannot have a
                                    bigger distance then img_shape[1]/inv_max_dist.
        max_angle (float):          The maximum difference in angles lines can have within a group.
        all_groups (bool):          If there are more than 2 groups outputted after grouping, you can decide to use
                                    the first two groups or all groups. If set to True, you use all groups, and you keep
                                    grouping the lines together until only 2 groups are left (by increasing the max
                                    distance iteratively). If set to False, you only use the first two groups.

    Returns:
        mask_lines (ndarray):
    """
    if after_grouping is None:
        logging.warning("No lines were outputted from the line detection, so no line mask can be created.")
        return np.zeros(img_shape, dtype=bool)

    if len(after_grouping) == 0:
        logging.warning("No lines were outputted from the line detection, so no line mask can be created.")
        return np.zeros(img_shape, dtype=bool)

    x_lines3 = x_lines2[after_grouping]
    y_lines3 = y_lines2[after_grouping]
    lines3 = lines2[after_grouping]
    angle_lines3 = angle_lines2[after_grouping]

    groups2 = group_single_lines(x_lines3, y_lines3, lines3, angle_lines3,
                                 max_distance=img_shape[1] / inv_max_dist, max_angle_diff=max_angle)

    if all_groups:
        while len(groups2) > 2:
            inv_max_dist -= 1
            groups2 = group_single_lines(x_lines3, y_lines3, lines3, angle_lines3,
                                         max_distance=img_shape[1] / inv_max_dist, max_angle_diff=max_angle)

    print(len(groups2))
    if len(groups2) > 1:
        x_left_mean = np.mean(x_lines3[groups2[0]])
        y_left_mean = np.mean(y_lines3[groups2[0]])
        angle_left_mean = np.mean(angle_lines3[groups2[0]])
        x_right_mean = np.mean(x_lines3[groups2[1]])
        y_right_mean = np.mean(y_lines3[groups2[1]])
        angle_right_mean = np.mean(angle_lines3[groups2[1]])
        # tan(a)*(x-x_mean) = y-y_mean

        x_grid = np.arange(0, img_shape[1], 1)
        y_grid = np.arange(0, img_shape[0], 1)

        mask_lines = np.zeros(img_shape, dtype=bool)
        for y in y_grid:
            x_line_left = (y - y_left_mean) / np.tan(angle_left_mean) + x_left_mean
            x_line_right = (y - y_right_mean) / np.tan(angle_right_mean) + x_right_mean
            x_line_left = np.ones(mask_lines.shape[1]) * x_line_left
            x_line_right = np.ones(mask_lines.shape[1]) * x_line_right
            if x_line_left[0] < x_line_right[0]:
                mask_lines[y, :] = (x_line_left < x_grid) & (x_grid < x_line_right)
            else:
                mask_lines[y, :] = (x_line_right < x_grid) & (x_grid < x_line_left)

        return mask_lines

    else:
        logging.warning("The lines were wrongly grouped in the creating of the mask in create_line_mask().")
        return None


def combine_masks(mask_diff, mask_lines, mask_lines_all, thres_diff=70, thres_lines=5, y_cut_off=6, it=75,
                  squaring=True):
    """
    Here we look if the mask from the difference in intensity and the mask from the line detection can be combined.
    The overlap between the masks is used as a measure to see if they can be combined or not.
    If possible, the masks are combined and binary dilated with the mask lines as constraint.
    If not the mask from the difference is used and optionally squared.

    Parameters:
        mask_diff (ndarray):        The mask from create_diff_mask().
        mask_lines (ndarray):       The mask from create_line_mask() with only the first 2 groups.
        mask_lines_all (ndarray):   The mask from create_line_mask() with all the groups.
        thres_diff (int):           The minimal % in overlap for the mask_diff.
        thres_lines (int):          The minimal % in overlap for the mask_lines.
        y_cut_off (int):            mask[:mask.shape[1]/y_cut_off, :] = False
                                    mask[-mask.shape[1]/y_cut_off:, :] = False
        it (int):                   How much you want the combined mask to be dilated in the y-direction
                                    (the x-direction will be fully filled to the mask_lines constraint).
        squaring (bool):            If the masks cannot be combined, set to True if you want the mask_diff to be a
                                    square. It will not square if the mask is too big which indicates that there are
                                    probably multiple milling sites or the masking went wrong.

    Returns:
        mask_combined (ndarray):    The masks combined if possible or the mask_diff.
        combined (bool):            A boolean that indicates if the masks are combined or not.
    """

    mask_combined = mask_lines * mask_diff
    mask_combined_all = mask_lines_all * mask_diff

    if np.sum(mask_lines) > 0:
        overlap_lines = np.sum(mask_combined) / np.sum(mask_lines) * 100
    else:
        overlap_lines = 0

    if np.sum(mask_diff) > 0:
        overlap_diff = np.sum(mask_combined) / np.sum(mask_diff) * 100
    else:
        overlap_diff = 0

    if np.sum(mask_lines_all) > 0:
        overlap_lines_all = np.sum(mask_combined_all) / np.sum(mask_lines_all) * 100
    else:
        overlap_lines_all = 0

    if np.sum(mask_diff) > 0:
        overlap_diff_all = np.sum(mask_combined_all) / np.sum(mask_diff) * 100
    else:
        overlap_diff_all = 0

    if overlap_diff_all > overlap_diff:
        mask_combined = mask_combined_all
        overlap_diff = overlap_diff_all
        overlap_lines = overlap_lines_all

    # print("The overlap with mask diff is {}%.".format(overlap_diff))
    # print("The overlap with mask lines is {}%.".format(overlap_lines))
    # print("The line mask will be used : {}".format((overlap_diff > thres_diff) | (overlap_lines > thres_lines)))
    if (overlap_diff > thres_diff) | (overlap_lines > thres_lines):
        mask_combined[:int(mask_combined.shape[0] / y_cut_off), :] = False
        mask_combined[-int(mask_combined.shape[0] / y_cut_off):, :] = False

        # squaring the mask
        index_mask = np.where(mask_combined)

        if len(index_mask[0]) != 0:
            x_min = np.min(index_mask[1])
            y_min = np.min(index_mask[0])
            x_max = np.max(index_mask[1])
            y_max = np.max(index_mask[0])
            mask_combined[y_min:y_max, x_min:x_max] = True

        # dilate the mask to the boundary lines
        mask_combined = binary_dilation(mask_combined, iterations=it) * mask_lines
        mask_combined = binary_dilation(mask_combined, structure=np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]),
                                        iterations=0, mask=mask_lines)
        mask_combined = binary_erosion(mask_combined, structure=np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]),
                                       iterations=10)
        return mask_combined, True
    else:
        mask_diff[:int(mask_diff.shape[0] / y_cut_off), :] = False
        mask_diff[-int(mask_diff.shape[0] / y_cut_off):, :] = False
        index_mask = np.where(mask_diff)

        # squaring the mask
        if squaring & (len(index_mask[0]) != 0):
            x_min = np.min(index_mask[1])
            y_min = np.min(index_mask[0])
            x_max = np.max(index_mask[1])
            y_max = np.max(index_mask[0])
            # if the mask is too big, then there are maybe multiple milling sites so squaring will not be useful
            if x_max - x_min <= mask_diff.shape[1] / 3:
                mask_diff[y_min:y_max, x_min:x_max] = True
        else:
            mask_diff = binary_dilation(mask_diff, iterations=10)

        return mask_diff, False


def create_masked_img(img_after, mask, cropping=False):
    """
    Multiplies the mask with the image after milling, so you only see the signal within the mask.
    If cropping set to True and there is a mask, the outputted image is cropped to show only the mask.

    Parameters:
        img_after (ndarray):    The image to be multiplied with the mask.
        mask (ndarray):         The mask to be multiplied with the image.
        cropping (bool):        If set to true, the image is cropped to show only the mask.

    Returns:
        masked_img (ndarray):   The image signal at the mask.
        extents (tuple):        The extent of the cropped image. (for plotting and visualization purposes)
    """

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


def create_binary_end_image(mask, masked_img, threshold=0.25, open_close=True, rid_of_back_signal=True):
    """
    To apply a threshold on the image signal within the mask.
    After the threshold, some binary closing, opening and again closing is performed (optional but recommended).
    The result is thought to be signal instead of noise.

    Parameters:
        mask (ndarray):                 The mask with which the masked_img was made.
        masked_img (ndarray):           The image signal within the mask (output from create_masked_img).
        threshold (float):              The threshold for signal detection (just a simple img >= threshold).
        open_close (bool):              If the binary operations should be executed or not.
        rid_of_back_signal (bool):      All the blobs touching the boundaries, will be removed.

    Returns:
        binary_end_result (ndarray):    A binary image which shows the thought to be signal in the masked image.
    """

    binary_end_result = masked_img >= threshold

    # getting rid of too small ROIs and noise after the threshold
    if open_close:
        binary_end_result = binary_closing(binary_opening(binary_closing(binary_end_result, iterations=1),
                                                          iterations=4), iterations=6)
    if rid_of_back_signal:
        index_mask = np.where(mask)
        if len(index_mask[0]) != 0:
            x_min = np.min(index_mask[1])
            y_min = np.min(index_mask[0])
            x_max = np.max(index_mask[1])
            y_max = np.max(index_mask[0])
            mask = mask[y_min:y_max, x_min:x_max]
        bound = binary_end_result * (1.0 * mask - 1.0 * binary_erosion(mask, iterations=10))

        minus = binary_dilation(bound, mask=binary_end_result, iterations=0) * binary_end_result
        binary_end_result_without = 1.0 * binary_end_result - 1.0 * minus
        binary_end_result_without = np.array(binary_end_result_without, dtype=bool)
    else:
        binary_end_result_without = binary_end_result

    return binary_end_result, binary_end_result_without


def detect_blobs(binary_end_result2, min_circ=0, max_circ=1, min_area=0, max_area=np.inf, min_in=0, max_in=1,
                 plotting=False):
    """
    Here blob detection from opencv is performed on a binary image (or an image with intensities ranging from 0 to 1).
    You can set the constraints of the blob detection with the parameters of this function.

    If you want more information on the opencv blob detection and how the parameters exactly work:
    https://learnopencv.com/blob-detection-using-opencv-python-c/

    Parameters:
        binary_end_result2 (ndarray):   The image on which blob detection should be performed.
        min_circ (float):               The minimal circularity the blob needs to have to be detected (range:[0,1])
        max_circ (float):               The maximal circularity the blob needs to have to be detected (range:[0,1])
        min_area (float):               The minimal area in pixels the blob needs to have to be detected (range:[0,inf])
        max_area (float):               The maximal area in pixels the blob needs to have to be detected (range:[0,inf])
        min_in (float):                 The minimal inertia the blob needs to have to be detected (range:[0,1])
        max_in (float):                 The maximal inertia the blob needs to have to be detected (range:[0,1])
        plotting (bool):                If set to True, the detected blobs are shown with red circles in the image.

    Returns:
        key_points (tuple):             The raw output of the opencv blob detection.
        yxr (ndarray):                  Only the y,x position and the radius of the blob in a numpy array.

    """

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
    """
    This functions shows 6 images on different processing steps in this script. It shows the input used in the very
    beginning, where the mask is found, and what the binary signal is within the mask.

    Parameters:
         img_before (ndarray):          The image before milling.
         img_after (ndarray):           The image after milling.
         img_before_blurred (ndarray):  The image before milling normalized and blurred.
         img_after_blurred (ndarray):   The image after milling normalized and blurred.
         mask (ndarray):                The mask of the milling site.
         masked_img (ndarray):          The image signal within the mask without binary operations.
         masked_img2 (ndarray):         The image signal within the mask with binary operations.
         binary_end_result1 (ndarray):  The binary end result of the threshold on the masked_img.
         binary_end_result2 (ndarray):  The binary end result of the threshold on the masked_img2.
         cropping (bool):               Boolean on if the image was cropped or not around the mask.
         extents (tuple):               The extents on the cropped (or un-cropped) masked_img.
         extents2 (tuple):              The extents on the cropped (or un-cropped) masked_img2.

    Returns:
        Nothing, only shows the images with matplotlib.pyplot.
    """

    # plotting the results
    x_min = extents[0]
    x_max = extents[1]
    y_min = extents[3]
    y_max = extents[2]
    img_before_blurred = deepcopy(img_after_blurred)
    ints_before = np.max(img_before_blurred) * 1.1
    ints_after = np.max(img_after_blurred) * 1.1
    cv2.line(img_before_blurred, (x_min, y_min), (x_max, y_min), ints_before, 5)
    cv2.line(img_before_blurred, (x_max, y_min), (x_max, y_max), ints_before, 5)
    cv2.line(img_before_blurred, (x_min, y_max), (x_max, y_max), ints_before, 5)
    cv2.line(img_before_blurred, (x_min, y_min), (x_min, y_max), ints_before, 5)
    x_min = extents2[0]
    x_max = extents2[1]
    y_min = extents2[3]
    y_max = extents2[2]
    cv2.line(img_after_blurred, (x_min, y_min), (x_max, y_min), ints_after, 5)
    cv2.line(img_after_blurred, (x_max, y_min), (x_max, y_max), ints_after, 5)
    cv2.line(img_after_blurred, (x_min, y_max), (x_max, y_max), ints_after, 5)
    cv2.line(img_after_blurred, (x_min, y_min), (x_min, y_max), ints_after, 5)

    fig, ax = plt.subplots(4, 2)  # , figsize=(15, 15))  # , sharey=True, sharex=True)  , dpi=600)

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

    # plt.savefig("/home/victoria/Documents/Lars/figures/data nr{} max ip.png".format(e), dpi=300)  # , dpi=600)
    plt.show()
    # del fig, ax
