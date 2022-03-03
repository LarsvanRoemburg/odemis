import numpy as np
import matplotlib.pyplot as plt
from odemis.dataio import tiff
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import binary_erosion, binary_dilation, binary_opening, binary_closing
from scipy.signal import fftconvolve
# from cv2 import HoughCircles, HOUGH_GRADIENT

# import analyse_shifts

# which_path = 4
threshold_mask = 0.3
threshold_end = 0.25
blur = 25
cropping = True
squaring = False
mean_zstack = False  # if true the mean is taken from the z-stack, if false the max intensity projection is taken
max_zstack = True - mean_zstack

channel_before = [0, 0, 0, 0, 0, 0, 0, 1, 0]
channel_after = [0, 0, 0, 0, 0, 0, 0, 0, 0]
manual_thr_out_before = [300, 700, 3000, 300, 300, 300, 300, 200, 750]  # make this percentage of max image intensity
manual_thr_out_after = [300, 700, 3000, 300, 300, 300, 300, 8000, 900]  # make this percentage of max image intensity
z_value_before = [20, 23, 20, 25, 30, np.pi, 25, np.pi, 27]  # np.pi means it is not a z-stack, only one image
z_value_after = [13, 15, 20, 19, 30, np.pi, 19, 1, 22]  # np.pi means it is not a z-stack, only one image

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
                     "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Mammalian_cells/FOV7_cp00_GFP.tif"
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
                    "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Mammalian_cells/FOV7_after_GFP.tif"
                    ]
for nnn in np.arange(3, 9, 1, dtype=int):
    print("dataset nr. {}".format(nnn+1))

    # convert the image to a numpy array and set a threshold for outliers in the image / z-stack
    data_before_milling = tiff.read_data(data_paths_before[nnn])
    meta_before = data_before_milling[channel_before[nnn]].metadata

    if z_value_before[nnn] != np.pi:
        img_before = np.array(data_before_milling[channel_before[nnn]][0][0], dtype=int)
        del data_before_milling

        histo, bins = np.histogram(img_before, bins=2000)
        histo = np.cumsum(histo)
        histo = histo / histo[-1]
        plc = np.min(np.where(histo > 0.99))
        thr_out_before = bins[plc] / 2 + bins[plc + 1] / 2
        img_before[img_before > thr_out_before] = thr_out_before

        if mean_zstack:
            img_before = np.mean(img_before, axis=0)
        else:
            img_before = np.max(img_before, axis=0)

    else:
        img_before = np.array(data_before_milling[channel_before[nnn]], dtype=int)
        del data_before_milling

        histo, bins = np.histogram(img_before, bins=2000)
        histo = np.cumsum(histo)
        histo = histo / histo[-1]
        plc = np.min(np.where(histo > 0.99))
        thr_out_before = bins[plc] / 2 + bins[plc + 1] / 2
        img_before[img_before > thr_out_before] = thr_out_before

    del histo, bins

    data_after_milling = tiff.read_data(data_paths_after[nnn])
    meta_after = data_after_milling[channel_after[nnn]].metadata

    if z_value_after[nnn] != np.pi:
        img_after = np.array(data_after_milling[channel_after[nnn]][0][0], dtype=int)
        del data_after_milling

        histo, bins = np.histogram(img_after, bins=2000)
        histo = np.cumsum(histo)
        histo = histo / histo[-1]
        plc = np.min(np.where(histo > 0.99))
        thr_out_after = bins[plc] / 2 + bins[plc + 1] / 2
        img_after[img_after > thr_out_after] = thr_out_after

        if mean_zstack:
            img_after = np.mean(img_after, axis=0)
        else:
            img_after = np.max(img_after, axis=0)
    else:
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

    # plotting a histogram of the intensity values of ...
    # plt.title('histogram')
    # plt.xlabel("value")
    # plt.ylabel("frequency")
    # plt.show()

    # rescaling one image to the other if necessary
    if img_after.shape != img_before.shape:
        if img_after.shape > img_before.shape:
            img_rescaled = np.zeros(img_after.shape)
            d_pix = img_before.shape[0] / img_after.shape[
                0]  # here we assume that the difference in x is the same as in
            # y as pixels are squares
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
            del x_vals, y_vals, x_new, y_new, img_rescaled

        else:
            img_rescaled = np.zeros(img_before.shape)
            d_pix = img_after.shape[0] / img_before.shape[
                0]  # here we assume that the difference in x is the same as in
            # y as pixels are squares
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

    if dx_pix > 0:
        img_after[:, dx_pix:] = img_after[:, :-dx_pix]
    elif dx_pix < 0:
        img_after[:, :dx_pix] = img_after[:, -dx_pix:]

    if dy_pix > 0:
        img_after[dy_pix:, :] = img_after[:-dy_pix, :]
    elif dy_pix < 0:
        img_after[:dy_pix, :] = img_after[-dy_pix:, :]

    # print(np.min(img_before))
    # print(np.min(img_after))
    # print("\n")
    # print(np.max(img_before))
    # print(np.max(img_after))
    # print("\n")

    # preprocessing steps of the images: blurring the image
    img_after_blurred = gaussian_filter(img_after, sigma=blur)
    img_before_blurred = gaussian_filter(img_before, sigma=blur)

    # print(np.min(img_before_blurred))
    # print(np.min(img_after_blurred))
    # print(np.max(img_before_blurred))
    # print(np.max(img_after_blurred))
    # print("\n")

    # preprocessing steps of the images: normalization to [0,1] interval
    base_lvl_before = np.min(img_before_blurred)
    base_lvl_after = np.min(img_after_blurred)
    img_before_blurred = (img_before_blurred - base_lvl_before) / (np.max(img_before_blurred) - base_lvl_before)
    img_after_blurred = (img_after_blurred - base_lvl_after) / (np.max(img_after_blurred) - base_lvl_after)

    img_before = (img_before - base_lvl_before) / (np.max(img_before) - base_lvl_before)
    img_after = (img_after - base_lvl_after) / (np.max(img_after) - base_lvl_after)
    img_before[img_before < 0] = 0
    img_after[img_after < 0] = 0

    # calculating the difference between the two images and creating a mask
    diff = img_before_blurred - 1.5*img_after_blurred  # times 2 to account for the intensity differences (more robust)
    mask = diff >= threshold_mask
    mask2 = binary_opening(mask, iterations=int(2*blur/3))  # First opening
    # mask2 = binary_erosion(mask2, iterations=5)  # if the edges give too much signal we can erode the mask a bit more

    # to crop the image around the ROI if possible and wanted: cropping = True
    index_mask = np.where(mask)
    index_mask2 = np.where(mask2)

    if squaring & (len(index_mask2[0]) != 0):
        x_min = np.min(index_mask2[1])
        y_min = np.min(index_mask2[0])
        x_max = np.max(index_mask2[1])
        y_max = np.max(index_mask2[0])
        mask2[y_min:y_max, x_min:x_max] = True

    masked_img = img_after * mask
    masked_img2 = img_after * mask2

    if cropping & (len(index_mask[0]) != 0):
        x_min = np.min(index_mask[1])
        y_min = np.min(index_mask[0])
        x_max = np.max(index_mask[1])
        y_max = np.max(index_mask[0])
    # else:
    #     x_min = 0
    #     y_min = 0
    #     x_max = img_after.shape[1]
    #     y_max = img_after.shape[0]

        masked_img = masked_img[y_min:y_max, x_min:x_max]
        masked_img2 = masked_img2[y_min:y_max, x_min:x_max]

    # setting a threshold for the image_after within the mask
    binary_end_result1 = masked_img >= threshold_end
    binary_end_result2 = masked_img2 >= threshold_end

    # getting rid of too small ROIs and noise after the threshold
    # binary_end_result1 = binary_opening(binary_end_result1, iterations=5)
    binary_end_result2 = binary_opening(binary_closing(binary_end_result2, iterations=2),
                                        iterations=4)  # second opening

    # circles = HoughCircles(binary_end_result2, HOUGH_GRADIENT, 1, 20, param1=60, param2=20,
    #                        minRadius=0, maxRadius=img_after.shape[0] / 4)

    # print(np.min(img_before))
    # print(np.min(img_after))
    # print(np.min(img_before_blurred))
    # print(np.min(img_after_blurred))
    # print(np.min(diff))
    # print("\n")
    # print(np.max(img_before))
    # print(np.max(img_after))
    # print(np.max(img_before_blurred))
    # print(np.max(img_after_blurred))
    # print(np.max(diff))

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
        shift, plc, dx_pix, dy_pix

    print("Successful!\n")