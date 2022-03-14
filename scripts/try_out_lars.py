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

# import analyse_shifts

threshold_mask = 0.3
threshold_end = 0.25
blur = 25
max_slices = 40
cropping = True  # if true, the last images will be cropped to only the mask
squaring = True  # if true, the mask will be altered to be a square
mean_z_stack = False  # if true the mean is taken from the z-stack, if false the max intensity projection is taken

data_paths_before = ["/home/victoria/Documents/Lars/data/1/FOV2_GFP_cp00.tif",
                     "/home/victoria/Documents/Lars/data/2/FOV1_checkpoint_00.tiff",
                     "/home/victoria/Documents/Lars/data/3/FOV4_checkpoint_00.tiff",
                     "/home/victoria/Documents/Lars/data/4/METEOR_images/FOV3_checkpoint_00.tif",
                     "/home/victoria/Documents/Lars/data/6/20201002_FOV2_checkpoint_001_stack_001.tiff",
                     "/home/victoria/Documents/Lars/data/7/FOV3_Meteor_stacks/Lamella_slices_tif"
                     "/EA010_8_FOV3_checkpoint_00_1-24.tif",
                     "/home/victoria/Documents/Lars/data/meteor1/FOV3_Meteor_stacks/EA010_8_FOV3_checkpoint_00_1.tiff",
                     "/home/victoria/Documents/Lars/data/XA Yeast-20220215T093027Z-001/XA "
                     "Yeast/20200918_millingsite_start.ome.tiff",
                     "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Mammalian_cells/FOV7_cp00_GFP.tif",
                     "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Mammalian_cells/FOV9_cp00_GFP.tif",
                     "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Mammalian_cells/FOV11_cp00_GFP.tif",
                     "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Mammalian_cells/FOV12_cp00_GFP.tif",
                     "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Yeast_eGFP-Ede1/G2_FOV1_checkpoint00.tif",
                     "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Yeast_eGFP-Ede1/G2_FOV2_checkpoint00.tif",
                     "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Yeast_eGFP-Ede1/G2_FOV3_checkpoint00.tif",
                     "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Yeast_eGFP-Ede1/G2_FOV4_checkpoint00.tif",
                     "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/negative_examples/yeast/"
                     "FOV3_new_checkpoint_00.tiff",
                     "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/negative_examples/yeast/"
                     "FOV6_checkpoint00_ch00.tiff"
                     ]

data_paths_after = ["/home/victoria/Documents/Lars/data/1/FOV2_GFP_cp04.tif",
                    "/home/victoria/Documents/Lars/data/2/FOV1_checkpoint_01.tiff",
                    "/home/victoria/Documents/Lars/data/3/FOV4_checkpoint_01.tiff",
                    "/home/victoria/Documents/Lars/data/4/METEOR_images/FOV3_checkpoint_01.tif",
                    "/home/victoria/Documents/Lars/data/6/20201002_FOV2_checkpoint_005_stack_001.tiff",
                    "/home/victoria/Documents/Lars/data/7/FOV3_Meteor_stacks/Lamella_slices_tif"
                    "/EA010_8_FOV3_final-19.tif",
                    "/home/victoria/Documents/Lars/data/meteor1/FOV3_Meteor_stacks/EA010_8_FOV3_final.tiff",
                    "/home/victoria/Documents/Lars/data/XA Yeast-20220215T093027Z-001/XA "
                    "Yeast/20200918-zstack_400nm.ome.tiff",
                    "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Mammalian_cells/FOV7_after_GFP.tif",
                    "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Mammalian_cells/FOV9_after_GFP.tif",
                    "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Mammalian_cells/FOV11_after_GFP.tif",
                    "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Mammalian_cells/FOV12_after_GFP.tif",
                    "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Yeast_eGFP-Ede1/G2_FOV1_checkpoint03.tif",
                    "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Yeast_eGFP-Ede1/G2_FOV2_checkpoint04.tif",
                    "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Yeast_eGFP-Ede1/G2_FOV3_checkpoint04.tif",
                    "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Yeast_eGFP-Ede1/G2_FOV4_checkpoint04.tif",
                    "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/negative_examples/yeast/"
                    "FOV3_new_final_lamella.tiff",
                    "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/negative_examples/yeast/"
                    "FOV6_checkpoint02_ch00.tiff"
                    ]

l = len(data_paths_before)
channel_before = np.zeros(l, dtype=int)
channel_before[7] = 1
channel_after = np.zeros(l, dtype=int)
manual_thr_out_before = np.zeros(l, dtype=int)
manual_thr_out_after = np.zeros(l, dtype=int)
manual_thr_out_before[:9] = np.array([300, 700, 3000, 300, 300, 300, 300, 200, 750])
manual_thr_out_after[:9] = np.array([300, 700, 3000, 300, 300, 300, 300, 8000, 900])

for nnn in np.arange(12, 18, 1, dtype=int):  # range(len(data_paths_after)) OR np.arange(4, 9, 1, dtype=int)
    print("dataset nr. {}".format(nnn + 1))
    print(data_paths_before[nnn])

    # convert the image to a numpy array and set a threshold for outliers in the image / z-stack
    # for data before milling
    data_before_milling = tiff.read_data(data_paths_before[nnn])
    meta_before = data_before_milling[channel_before[nnn]].metadata

    if data_before_milling[0].shape[0] == 1:
        num_z = len(data_before_milling[channel_before[nnn]][0][0])
        print("#z-slices before: {}".format(num_z))
        if num_z <= max_slices:  # if there are too many slices,
            # take the #max_slices in the middle
            img_before = np.array(data_before_milling[channel_before[nnn]][0][0], dtype=int)
        else:
            s = int((num_z - max_slices) / 2)
            img_before = np.array(data_before_milling[channel_before[nnn]][0][0][s:-s], dtype=int)
        del data_before_milling

        histo, bins = np.histogram(img_before, bins=2000)
        histo = np.cumsum(histo)
        histo = histo / histo[-1]
        plc = np.min(np.where(histo > 0.99))
        thr_out_before = bins[plc] / 2 + bins[plc + 1] / 2
        img_before[img_before > thr_out_before] = thr_out_before

        if mean_z_stack:
            img_before = np.mean(img_before, axis=0)
        else:
            img_before = np.max(img_before, axis=0)

    else:
        print("#z-slices before: 1")
        img_before = np.array(data_before_milling[channel_before[nnn]], dtype=int)
        del data_before_milling

        histo, bins = np.histogram(img_before, bins=2000)
        histo = np.cumsum(histo)
        histo = histo / histo[-1]
        plc = np.min(np.where(histo > 0.99))
        thr_out_before = bins[plc] / 2 + bins[plc + 1] / 2
        img_before[img_before > thr_out_before] = thr_out_before

    del histo, bins

    # for data after milling
    data_after_milling = tiff.read_data(data_paths_after[nnn])
    meta_after = data_after_milling[channel_after[nnn]].metadata

    if data_after_milling[0].shape[0] == 1:
        print("#z-slices after: {}".format(len(data_after_milling[channel_after[nnn]][0][0])))
        if len(data_after_milling[channel_after[nnn]][0][0]) <= max_slices:  # if there are too many slices,
            # take the #max_slices in the middle
            img_after = np.array(data_after_milling[channel_after[nnn]][0][0], dtype=int)
        else:
            s = int((len(data_after_milling[channel_after[nnn]][0][0]) - max_slices) / 2)
            img_after = np.array(data_after_milling[channel_after[nnn]][0][0][s:-s], dtype=int)
        del data_after_milling

        histo, bins = np.histogram(img_after, bins=2000)
        histo = np.cumsum(histo)
        histo = histo / histo[-1]
        plc = np.min(np.where(histo > 0.99))
        thr_out_after = bins[plc] / 2 + bins[plc + 1] / 2
        img_after[img_after > thr_out_after] = thr_out_after

        if mean_z_stack:
            img_after = np.mean(img_after, axis=0)
        else:
            img_after = np.max(img_after, axis=0)
    else:
        print("#z-slices after: 1")
        img_after = np.array(data_after_milling[channel_after[nnn]], dtype=int)
        del data_after_milling

        histo, bins = np.histogram(img_after, bins=2000)
        histo = np.cumsum(histo)
        histo = histo / histo[-1]
        plc = np.min(np.where(histo > 0.99))
        thr_out_after = bins[plc] / 2 + bins[plc + 1] / 2
        img_after[img_after > thr_out_after] = thr_out_after

    del histo, bins

    print("Pixel sizes are the same: {}".format(meta_before["Pixel size"][0] == meta_after["Pixel size"][0]))
    print("manual outlier threshold: \nbefore: {}\nafter: {}".format(manual_thr_out_before[nnn],
                                                                     manual_thr_out_after[nnn]))
    print("histogram 99% threshold: \nbefore: {}\nafter: {}".format(thr_out_before, thr_out_after))

    # rescaling one image to the other if necessary
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
            img_before = img_rescaled

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
            img_after = img_rescaled
        del x_vals, y_vals, x_new, y_new, img_rescaled

    # calculate the shift between the two images
    mean_before = np.mean(img_before)
    mean_after = np.mean(img_after)
    img_before = img_before - mean_before
    img_after = img_after - mean_after
    conv = fftconvolve(img_before, img_after[::-1, ::-1], mode='same')
    conv[:int(conv.shape[0] / 5), :] = 0  # constraints
    conv[-int(conv.shape[0] / 5):, :] = 0
    conv[:, :int(conv.shape[1] / 5)] = 0
    conv[:, -int(conv.shape[1] / 5):] = 0

    shift = np.where(conv == np.max(conv))
    shift = np.asarray(shift)
    shift[0] = shift[0] - img_after.shape[0] / 2
    shift[1] = shift[1] - img_after.shape[1] / 2
    del conv
    img_before = img_before + mean_before
    img_after = img_after + mean_after
    # shift2 = analyse_shifts.measure_shift(im1, im2, use_md=False)

    dy_pix = int(shift[0])
    dx_pix = int(shift[1])

    print("dx is {} pixels".format(dx_pix))
    print("dy is {} pixels".format(dy_pix))

    # shifting the after milling image towards the before milling image to overlap nicely
    if dx_pix > 0:
        img_after[:, dx_pix:] = img_after[:, :-dx_pix]
    elif dx_pix < 0:
        img_after[:, :dx_pix] = img_after[:, -dx_pix:]

    if dy_pix > 0:
        img_after[dy_pix:, :] = img_after[:-dy_pix, :]
    elif dy_pix < 0:
        img_after[:dy_pix, :] = img_after[-dy_pix:, :]
        # img_after[dy_pix:, :] = np.min(img_after[dy_pix:, :])

    # preprocessing steps of the images: blurring the image
    img_after_blurred = gaussian_filter(img_after, sigma=blur)
    img_before_blurred = gaussian_filter(img_before, sigma=blur)

    # preprocessing steps of the images: normalization of blurred images to [0,1] interval
    base_lvl_before = np.min(img_before_blurred)
    base_lvl_after = np.min(img_after_blurred)
    img_before_blurred = (img_before_blurred - base_lvl_before) / (np.max(img_before_blurred) - base_lvl_before)
    img_after_blurred = (img_after_blurred - base_lvl_after) / (np.max(img_after_blurred) - base_lvl_after)

    # preprocessing steps of the images: normalization of initial images to [0,1] interval
    img_before = (img_before - base_lvl_before) / (np.max(img_before) - base_lvl_before)
    img_after = (img_after - base_lvl_after) / (np.max(img_after) - base_lvl_after)
    if dx_pix > 0:
        img_after[:, :dx_pix] = 0
        img_after_blurred[:, :dx_pix] = 0
    elif dx_pix < 0:
        img_after[:, dx_pix:] = 0
        img_after_blurred[:, dx_pix:] = 0
    if dy_pix > 0:
        img_after[:dy_pix, :] = 0
        img_after_blurred[:dy_pix, :] = 0
    elif dy_pix < 0:
        img_after[dy_pix:, :] = 0
        img_after_blurred[dy_pix:, :] = 0
    # you take base_lvl of blurred image because of noise in the initial image (the range would be bigger than needed)
    img_before[img_before < 0] = 0
    img_after[img_after < 0] = 0

    # detecting two parallel lines
    # the dip in this projection is where there is milled
    projection = np.sum(img_after_blurred, axis=0)

    # here I try line detection
    image = deepcopy(img_after_blurred)
    image = (image > 0.1)*0.1
    # image[image > np.max(image) / 3] = np.max(image) / 3

    image = img_as_ubyte(image)
    plt.imshow(image)
    edges = img_as_ubyte(canny(image, sigma=5, low_threshold=1, high_threshold=3))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=1, minLineLength=img_after.shape[1] / 10,
                            maxLineGap=img_after.shape[1] / 7.5)  # /7.5

    # Iterate over points
    if lines is not None:
        print("{} lines detected.".format(len(lines)))
        # look at the lines and see if they have the same angle and rough distance between them
        angle_lines = np.zeros(lines.shape[0])
        for i in range(lines.shape[0]):
            angle_lines[i] = np.arctan((lines[i][0][3] - lines[i][0][1]) / (lines[i][0][2] - lines[i][0][0]))

        diff_angles = np.zeros((lines.shape[0], lines.shape[0]), dtype=bool)
        for i in range(lines.shape[0]):
            for j in np.arange(i + 1, lines.shape[0], 1):
                diff_angles[i, j] = np.abs(angle_lines[i] - angle_lines[j]) < np.pi/32
                if diff_angles[i, j]:
                    x1 = lines[i][0][2] / 2 + lines[i][0][0] / 2
                    y1 = lines[i][0][3] / 2 + lines[i][0][1] / 2
                    x2 = lines[j][0][2] / 2 + lines[j][0][0] / 2
                    y2 = lines[j][0][3] / 2 + lines[j][0][1] / 2
                    mean_angle = (angle_lines[i] + angle_lines[j]) / 2
                    x3 = (y2 - y1 + np.tan(mean_angle + np.pi / 2 + 1e-10) * x1 - np.tan(mean_angle + 1e-10) * x2) / (
                                np.tan(mean_angle + np.pi / 2 + 1e-10) - np.tan(mean_angle + 1e-10))
                    y3 = np.tan(mean_angle + 1e-10) * (x3 - x2) + y2
                    dist = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
                    if not ((dist < img_after.shape[0]/5) & (dist > img_after.shape[0]/10)):
                        diff_angles[i, j] = False

        close_angles = np.where(diff_angles)
        # line_image = deepcopy(image/10)
        for points in lines[np.unique(close_angles)]:
            # Extracted points nested in the list
            x1, y1, x2, y2 = points[0]
            # Draw the lines joing the points
            # On the original image
            cv2.line(image, (x1, y1), (x2, y2), (255), 2)
            # Maintain a simples lookup list for points
            # lines_list.append([(x1, y1), (x2, y2)])
        plt.imshow(image)
    else:
        print("No lines are detected")

    # calculating the difference between the two images and creating a mask
    diff = img_before_blurred - 1.5 * img_after_blurred  # times 1.5 to account for intensity differences (more robust)
    mask = diff >= threshold_mask
    mask2 = binary_opening(mask, iterations=int(blur / 2))  # First opening
    # mask2 = binary_erosion(mask2, iterations=5)  # if the edges give too much signal we can erode the mask a bit more

    # to crop and/or square the image around the ROI if possible and wanted: cropping = True and/or squaring = True
    index_mask = np.where(mask)
    index_mask2 = np.where(mask2)

    # squaring the mask
    if squaring & (len(index_mask2[0]) != 0):
        x_min = np.min(index_mask2[1])
        y_min = np.min(index_mask2[0])
        x_max = np.max(index_mask2[1])
        y_max = np.max(index_mask2[0])
        mask2[y_min:y_max, x_min:x_max] = True

    mask = mask2

    # getting the after milling signal only in the mask
    masked_img = img_after * mask
    masked_img2 = img_after * mask2

    # cropping the image to the mask
    if cropping & (len(index_mask[0]) != 0):
        x_min = np.min(index_mask[1])
        y_min = np.min(index_mask[0])
        x_max = np.max(index_mask[1])
        y_max = np.max(index_mask[0])

        masked_img = masked_img[y_min:y_max, x_min:x_max]
        masked_img2 = masked_img2[y_min:y_max, x_min:x_max]
    else:  # for the extent in the plotting
        x_min = 0
        y_min = 0
        x_max = img_after.shape[1]
        y_max = img_after.shape[0]

    # setting a threshold for the image_after within the mask
    binary_end_result1 = masked_img >= threshold_end
    binary_end_result2 = masked_img2 >= threshold_end

    # getting rid of too small ROIs and noise after the threshold
    # binary_end_result1 = binary_dilation(binary_opening(binary_closing(binary_end_result1, iterations=1),
    #                                                 iterations=4), iterations=4)  # first closing and second opening
    binary_end_result2 = binary_dilation(binary_opening(binary_closing(binary_end_result2, iterations=1),
                                                        iterations=4), iterations=4)  # first closing and second opening
    print("something")

    # here I start with trying to detect circles
    im = np.array(binary_end_result2 * 255,
                  dtype=np.uint8)  # cv2.cvtColor(binary_end_result2.astype('uint8'), cv2.COLOR_BGR2GRAY)
    im = cv2.cvtColor(im, cv2.IMREAD_GRAYSCALE)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = False
    params.filterByConvexity = False
    params.filterByCircularity = True
    params.filterByArea = True
    params.filterByInertia = True
    params.minCircularity = 0.4
    params.minArea = 6 ** 2
    params.minInertiaRatio = 0.5
    # params.maxArea = binary_end_result2.shape[0] * binary_end_result2.shape[1] / 4
    # print(binary_end_result2.shape[0] * binary_end_result2.shape[1] / 4)
    ver = cv2.__version__.split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    # detector = cv2.SimpleBlobDetector_create()
    # im = cv2.bitwise_not(im)

    key_points = detector.detect(im)
    # im = cv2.bitwise_not(im)

    cx = np.zeros(len(key_points), dtype=int)
    cy = np.zeros(len(key_points), dtype=int)
    rr = np.zeros(len(key_points), dtype=int)
    for u in range(len(key_points)):
        cx[u] = int(key_points[u].pt[0])
        cy[u] = int(key_points[u].pt[1])
        rr[u] = int(key_points[u].size / 2)  # the size is the diameter

    # Draw them
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    for center_y, center_x, radius in zip(cy, cx, rr):
        circy, circx = circle_perimeter(center_y, center_x, radius,
                                        shape=image.shape)
        im[circy, circx] = (220, 20, 20, 255)

    ax.imshow(im, cmap=plt.cm.gray)
    plt.show()

    # im_with_keypoints = cv2.drawKeypoints(im, key_points, np.array([]), (0, 0, 255),
    #                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # # Show keypoints
    # cv2.imshow("Keypoints", im_with_keypoints)
    # cv2.waitKey(0)
    print("keypoints is done")

    # circles = cv2.HoughCircles(binary_end_result2.astype('uint8'), cv2.HOUGH_GRADIENT_ALT, 1, 20, param1=1, param2=0.1,
    #                            minRadius=0, maxRadius=int(binary_end_result2.shape[1] / 2))
    # # circles is in the format (x, y, r)
    #
    # # Load picture and detect edges
    # image = img_as_ubyte(binary_end_result2)
    # edges = canny(image, sigma=0, low_threshold=0, high_threshold=10)
    #
    # # Detect two radii
    # hough_radii = np.arange(5, int(binary_end_result2.shape[1] / 4), 5)
    # hough_res = hough_circle(edges, hough_radii)
    #
    # # Select the most prominent 3 circles
    # accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=10)
    #
    # # Draw them
    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    # image = color.gray2rgb(img_as_ubyte(edges))
    # for center_y, center_x, radius in zip(cy, cx, radii):
    #     circy, circx = circle_perimeter(center_y, center_x, radius,
    #                                     shape=image.shape)
    #     image[circy, circx] = (220, 20, 20)
    #
    # ax.imshow(image, cmap=plt.cm.gray)
    # plt.show()

    # print("something")
    # if circles is not None:
    #     print("there are {} circles detected".format(len(circles[0])))
    #     binary_end_result2 = np.zeros(binary_end_result2.shape)
    #     x_grid = np.arange(0, binary_end_result2.shape[1], 1)
    #     y_grid = np.arange(0, binary_end_result2.shape[0], 1)
    #     xy_grid = np.meshgrid(x_grid, y_grid)
    #     for (x, y, r) in circles[0]:
    #         circ = (xy_grid[1] - y) ** 2 + (xy_grid[0] - x) ** 2 <= r ** 2
    #         binary_end_result2 = binary_end_result2 + circ
    # else:
    #     print("there are no circles detected")

    masked_img[0, 0] = 1  # to give them the same color scale
    masked_img2[0, 0] = 1  # to give them the same color scale

    # plotting the results
    fig, ax = plt.subplots(4, 2)
    ax[0, 0].imshow(img_before)
    ax[0, 0].set_title("before")
    ax[0, 1].imshow(img_after)
    ax[0, 1].set_title("after")
    ax[1, 0].imshow(img_before_blurred)
    ax[1, 1].imshow(img_after_blurred)
    if cropping & (len(index_mask[0]) != 0):
        ax[2, 0].imshow(masked_img, extent=(x_min, x_max, y_max, y_min))
        ax[2, 1].imshow(masked_img2, extent=(x_min, x_max, y_max, y_min))
        ax[3, 0].imshow(binary_end_result1, extent=(x_min, x_max, y_max, y_min))
        ax[3, 1].imshow(binary_end_result2, extent=(x_min, x_max, y_max, y_min))
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

    del img_before, img_before_blurred, img_after, img_after_blurred, diff, mask, mask2, masked_img, masked_img2, \
        binary_end_result1, binary_end_result2, meta_before, meta_after, index_mask, index_mask2, fig, ax, \
        shift, plc, dx_pix, dy_pix, x_min, x_max, y_min, y_max  # , circles, image
    gc.collect()

    print("Successful!\n")
