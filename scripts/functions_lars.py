import numpy as np
import matplotlib.pyplot as plt
from odemis.dataio import tiff
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import binary_erosion, binary_dilation, binary_opening, binary_closing
from scipy.signal import fftconvolve
import cv2
import gc
from skimage import color  # , data
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from copy import deepcopy


def z_projection_and_outlier_cutoff(data, max_slices, which_channel=0, mode='max'):
    if data[0].shape[0] == 1:
        num_z = len(data[which_channel][0][0])
        print("#z-slices before: {}".format(num_z))
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
            print("mode input not recognized, maximum intensity projection is used.")
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
    if img_after.shape != img_before.shape:
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

        else:  # copied the code but then for the images swapped, this is better memory wise (I think)
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
    else:
        return img_before, img_after


def overlay(img_before, img_after, max_shift=5):
    # max shift is between 1 and inf, the higher, the higher the shift can be
    # if it is 1 no constraints will be given on the shift
    mean_after = np.mean(img_after)
    img_before = img_before - np.mean(img_before)
    img_after = img_after - mean_after
    conv = fftconvolve(img_before, img_after[::-1, ::-1], mode='same')

    if (int(conv.shape[0] / max_shift) > 0) & max_shift > 1:  # constraints
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

    yxr = np.zeros((len(key_points), 3))
    for u in range(len(key_points)):
        yxr[u, 1] = int(key_points[u].pt[0])
        yxr[u, 0] = int(key_points[u].pt[1])
        yxr[u, 2] = int(key_points[u].size / 2)  # the size is the diameter

    # Draw them
    if plotting:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
        for center_y, center_x, radius in zip(cy, cx, rr):
            circy, circx = circle_perimeter(center_y, center_x, radius,
                                            shape=im.shape)
            im[circy, circx] = (220, 20, 20, 255)
    
        ax.imshow(im)
        plt.show()
        
    return key_points, yxr


def plot_end_results(img_before, img_after, img_before_blurred, img_after_blurred, mask, masked_img, masked_img2,
                     binary_end_result1, binary_end_result2, cropping, extents, extents2):
    # plotting the results
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
