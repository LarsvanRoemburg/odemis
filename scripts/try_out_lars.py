from functions_lars import *

from odemis.dataio import tiff
import gc
from skimage.feature import canny

from skimage.util import img_as_ubyte
from copy import deepcopy

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

ll = len(data_paths_before)
channel_before = np.zeros(ll, dtype=int)
channel_before[7] = 1
channel_after = np.zeros(ll, dtype=int)
# manual_thr_out_before = np.zeros(ll, dtype=int)
# manual_thr_out_after = np.zeros(ll, dtype=int)
# manual_thr_out_before[:9] = np.array([300, 700, 3000, 300, 300, 300, 300, 200, 750])
# manual_thr_out_after[:9] = np.array([300, 700, 3000, 300, 300, 300, 300, 8000, 900])

threshold_mask = 0.3
threshold_end = 0.25
blur = 25
max_slices = 40
cropping = True  # if true, the last images will be cropped to only the mask
squaring = True  # if true, the mask will be altered to be a square

for nnn in np.arange(13, 18, 1, dtype=int):  # range(len(data_paths_after)) OR np.arange(4, 9, 1, dtype=int)
    print("dataset nr. {}".format(nnn + 1))
    print(data_paths_before[nnn])

    # convert the image to a numpy array and set a threshold for outliers in the image / z-stack
    # for data before milling
    data_before_milling = tiff.read_data(data_paths_before[nnn])
    meta_before = data_before_milling[channel_before[nnn]].metadata

    img_before = z_projection_and_outlier_cutoff(data_before_milling, max_slices, channel_before[nnn], mode='max')
    del data_before_milling

    # for data after milling
    data_after_milling = tiff.read_data(data_paths_after[nnn])
    meta_after = data_after_milling[channel_after[nnn]].metadata

    img_after = z_projection_and_outlier_cutoff(data_after_milling, max_slices, channel_after[nnn], mode='max')
    del data_after_milling

    print("Pixel sizes are the same: {}".format(meta_before["Pixel size"][0] == meta_after["Pixel size"][0]))

    # rescaling one image to the other if necessary
    img_before, img_after = rescaling(img_before, img_after)

    # calculate the shift between the two images
    img_after = overlay(img_before, img_after, max_shift=4)  # max_shift between 1 and inf

    # preprocessing steps of the images: blurring the image
    img_before, img_after, img_before_blurred, img_after_blurred = blur_and_norm(img_before, img_after, blur=blur)

    # detecting two parallel lines
    # the dip in this projection is where there is milled
    projection_before_x = np.sum(img_before_blurred, axis=0)
    projection_after_x = np.sum(img_after_blurred, axis=0)
    diff_proj = projection_before_x - projection_after_x
    projection_before_y = np.sum(img_before_blurred, axis=1)
    projection_after_y = np.sum(img_after_blurred, axis=1)

    fig, ax = plt.subplots(nrows=2, ncols=3)
    ax[0, 0].plot(projection_before_x)
    ax[0, 1].plot(projection_after_x)
    ax[0, 2].plot(diff_proj)
    ax[1, 0].plot(projection_before_y)
    ax[1, 1].plot(projection_after_y)
    ax[1, 2].plot(projection_before_y - projection_after_y)
    # plt.show()
    mid_milling_site = np.where(diff_proj == np.max(diff_proj))[0][0]

    # here I try line detection
    image = deepcopy(img_after_blurred)
    # image = (image > 0.1)*0.1
    # image[image > np.max(image) / 2] = np.max(image) / 2
    image = img_as_ubyte(image)
    image = cv2.bitwise_not(image)
    # image[image < np.max(image)*4/5] = 0  # 4/5
    # plt.imshow(image)
    edges = img_as_ubyte(canny(image, sigma=10, low_threshold=0.1, high_threshold=2.5))  # 10, 0.1, 2.5
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=1, minLineLength=img_after.shape[1] / 45,
                            maxLineGap=img_after.shape[1] / 40)  # np.pi/180, 1, /30, /40
    x_lines = (lines.T[2] / 2 + lines.T[0] / 2).T.reshape((len(lines),))  # the middle points of the line
    y_lines = (lines.T[3] / 2 + lines.T[1] / 2).T.reshape((len(lines),))

    image = image * 0.8
    # Iterate over points
    if lines is not None:
        print("{} lines detected.".format(len(lines)))
        # look at the lines and see if they have the same angle and rough distance between them
        angle_lines = np.zeros(lines.shape[0])
        for i in range(lines.shape[0]):
            if lines[i][0][2] != lines[i][0][0]:
                angle_lines[i] = np.arctan((lines[i][0][3] - lines[i][0][1]) / (lines[i][0][2] - lines[i][0][0]))
                if angle_lines[i] < 0:
                    angle_lines[i] = angle_lines[i] + np.pi
            else:
                angle_lines[i] = np.pi / 2

        lines2 = deepcopy(lines)
        angle_lines2 = deepcopy(angle_lines)
        x_lines2 = deepcopy(x_lines)
        y_lines2 = deepcopy(y_lines)
        num_lines_present = np.ones(lines.shape[0])

        for i in range(lines2.shape[0]):
            x1 = x_lines2[i]
            y1 = y_lines2[i]
            if (angle_lines2[i] <= np.pi / 12) | (angle_lines2[i] >= np.pi * 11 / 12) | \
                    (x1 < (mid_milling_site - img_after.shape[1] / 8)) | \
                    (x1 > (mid_milling_site + img_after.shape[1] / 8)) | \
                    (y1 < img_after.shape[0] / 8) | (y1 > img_after.shape[0] * 7 / 8):
                num_lines_present[i] = 0
            else:
                for j in range(lines2.shape[0]):
                    if (i != j) & (num_lines_present[i] >= 1) & (num_lines_present[j] >= 1):
                        dist = np.sqrt((x_lines2[i] - x_lines2[j]) ** 2 + (y_lines2[i] - y_lines2[j]) ** 2)
                        diff_angle = np.abs(angle_lines2[i] - angle_lines2[j])
                        if (dist <= img_after.shape[0] / 30) & (diff_angle <= np.pi / 12):
                            if i < j:
                                lines2[i][0][0] = (lines2[i][0][0] * num_lines_present[i] + lines2[j][0][0] *
                                                   num_lines_present[j]) / (num_lines_present[i] + num_lines_present[j])
                                lines2[i][0][1] = (lines2[i][0][1] * num_lines_present[i] + lines2[j][0][1] *
                                                   num_lines_present[j]) / (num_lines_present[i] + num_lines_present[j])
                                lines2[i][0][2] = (lines2[i][0][2] * num_lines_present[i] + lines2[j][0][2] *
                                                   num_lines_present[j]) / (num_lines_present[i] + num_lines_present[j])
                                lines2[i][0][3] = (lines2[i][0][3] * num_lines_present[i] + lines2[j][0][3] *
                                                   num_lines_present[j]) / (num_lines_present[i] + num_lines_present[j])
                                angle_lines2[i] = (angle_lines2[i] * num_lines_present[i]
                                                   + angle_lines2[j] * num_lines_present[j]) / \
                                                  (num_lines_present[i] + num_lines_present[j])
                                x_lines2[i] = (x_lines2[i] * num_lines_present[i] + x_lines2[j] * num_lines_present[
                                    j]) / \
                                              (num_lines_present[i] + num_lines_present[j])
                                y_lines2[i] = (y_lines2[i] * num_lines_present[i] + y_lines2[j] * num_lines_present[
                                    j]) / \
                                              (num_lines_present[i] + num_lines_present[j])
                                num_lines_present[i] = num_lines_present[i] + num_lines_present[j]
                                num_lines_present[j] = 0
                            else:
                                lines2[j][0][0] = (lines2[i][0][0] * num_lines_present[i] + lines2[j][0][0] *
                                                   num_lines_present[j]) / (num_lines_present[i] + num_lines_present[j])
                                lines2[j][0][1] = (lines2[i][0][1] * num_lines_present[i] + lines2[j][0][1] *
                                                   num_lines_present[j]) / (num_lines_present[i] + num_lines_present[j])
                                lines2[j][0][2] = (lines2[i][0][2] * num_lines_present[i] + lines2[j][0][2] *
                                                   num_lines_present[j]) / (num_lines_present[i] + num_lines_present[j])
                                lines2[j][0][3] = (lines2[i][0][3] * num_lines_present[i] + lines2[j][0][3] *
                                                   num_lines_present[j]) / (num_lines_present[i] + num_lines_present[j])
                                angle_lines2[j] = (angle_lines2[i] * num_lines_present[i]
                                                   + angle_lines2[j] * num_lines_present[j]) / \
                                                  (num_lines_present[i] + num_lines_present[j])
                                x_lines2[j] = (x_lines2[i] * num_lines_present[i] + x_lines2[j] * num_lines_present[
                                    j]) / \
                                              (num_lines_present[i] + num_lines_present[j])
                                y_lines2[j] = (y_lines2[i] * num_lines_present[i] + y_lines2[j] * num_lines_present[
                                    j]) / \
                                              (num_lines_present[i] + num_lines_present[j])
                                num_lines_present[j] = num_lines_present[i] + num_lines_present[j]
                                num_lines_present[i] = 0

        lines2 = lines2[num_lines_present > 0]
        x_lines2 = x_lines2[num_lines_present > 0]
        y_lines2 = y_lines2[num_lines_present > 0]
        angle_lines2 = angle_lines2[num_lines_present > 0]
        # lines = lines2
        # x_lines = x_lines2
        # y_lines = y_lines2
        # angle_lines = angle_lines2

        # diff_angles = np.zeros((lines2.shape[0], lines2.shape[0]), dtype=bool)
        print("mid_milling_site = {}".format(mid_milling_site))
        print("lower x boundary = {}".format(mid_milling_site - img_after.shape[1] / 8))
        print("upper x boundary = {}".format(mid_milling_site + img_after.shape[1] / 8))
        # for i in range(lines2.shape[0]):
        #     for j in np.arange(i + 1, lines2.shape[0], 1):
        #         # print(angle_lines[i] - angle_lines[j])
        #         if np.abs(angle_lines2[i] - angle_lines2[j]) < np.pi / 16:
        #             x1 = x_lines2[i]
        #             y1 = y_lines2[i]
        #             x2 = x_lines2[j]
        #             y2 = y_lines2[j]
        #             # print("angle < np.pi/6 | angle > np.pi*5/6 is true")
        #             angle_mean = (angle_lines2[i] + angle_lines2[j]) / 2
        #             x3 = (y2 - y1 + np.tan(angle_mean + np.pi / 2 + 1e-10) * x1 - np.tan(angle_mean + 1e-10) * x2) / \
        #                  (np.tan(angle_mean + np.pi / 2 + 1e-10) - np.tan(angle_mean + 1e-10))
        #             y3 = np.tan(angle_mean + 1e-10) * (x3 - x2) + y2
        #             dist = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
        #             # print("distance measured = {}".format(dist))
        #             if (dist < img_after.shape[1] / 5) & (dist < img_after.shape[1] / 40):  # </5, >/20
        #                 diff_angles[i, j] = True
        #                 # print("pair is added")
        # print("minimal distance = {}".format(img_after.shape[1] / 20))
        # print("maximal distance = {}".format(img_after.shape[1] / 40))
        # close_angles = np.where(diff_angles)
        # print("{} lines after selection".format(len(np.unique(close_angles))))
        # lin1 = close_angles[0]  # the "left" line of the pair
        # lin2 = close_angles[1]  # the "right" line of the pair

        groups = []
        # for i in range(len(lin1)):
        #     set_in_group = False
        #     x1 = x_lines[lin1[i]]
        #     y1 = y_lines[lin1[i]]
        #     angle = angle_lines[lin1[i]]
        #     for j in range(len(groups)):
        #         angle_mean = np.mean(angle_lines[groups[j]])
        #         x2 = np.mean(x_lines[groups[j]])
        #         y2 = np.mean(y_lines[groups[j]])
        #         if np.abs(angle - angle_mean) <= np.pi/6:
        #             x3 = (y2 - y1 + np.tan(angle_mean + np.pi / 2 + 1e-10) * x1 - np.tan(angle_mean + 1e-10) * x2) \
        #                  / (np.tan(angle_mean + np.pi / 2 + 1e-10) - np.tan(angle_mean + 1e-10))
        #             y3 = np.tan(angle_mean + 1e-10) * (x3 - x2) + y2
        #             dist = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
        #             if dist <= img_after.shape[1]/20:
        #                 set_in_group = True
        #                 groups[j].append(lin1[i])
        #     if not set_in_group:
        #         groups.append([lin1[i]])
        #
        # for i in range(len(groups)):
        #     groups[i] = np.unique(groups[i])
        # groups_combined = np.zeros((len(groups), len(groups)), dtype=bool)
        #
        # for i in range(len(groups)):
        #     for j in np.arange(i+1, len(groups), 1):
        #         x1 = np.mean(x_lines[groups[i]])
        #         y1 = np.mean(y_lines[groups[i]])
        #         x2 = np.mean(x_lines[groups[j]])
        #         y2 = np.mean(y_lines[groups[j]])
        #         angle_1 = np.mean(angle_lines[groups[i]])
        #         angle_2 = np.mean(angle_lines[groups[j]])
        #         if np.abs(angle - angle_2) <= np.pi/6:
        #             x3 = (y2 - y1 + np.tan(angle_2 + np.pi / 2 + 1e-10) * x1 - np.tan(angle_2 + 1e-10) * x2) \
        #                  / (np.tan(angle_2 + np.pi / 2 + 1e-10) - np.tan(angle_2 + 1e-10))
        #             y3 = np.tan(angle_2 + 1e-10) * (x3 - x2) + y2
        #             dist = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
        #             if (dist > img_after.shape[1]/20) & (dist < img_after.shape[1]/4):
        #                 groups_combined[i, j] = True

        for i in range(len(lines2)):
            # print("{}% done".format(i / len(lines2) * 100))
            set_in_group = False
            x1 = x_lines2[i]
            y1 = y_lines2[i]
            angle = angle_lines2[i]
            for j in range(len(groups)):
                angle_mean = np.mean(angle_lines2[groups[j]])
                x2 = np.mean(x_lines2[groups[j]])
                y2 = np.mean(y_lines2[groups[j]])
                if np.abs(angle - angle_mean) <= np.pi / 16:
                    x3 = (y2 - y1 + np.tan(angle_mean + np.pi / 2 + 1e-10) * x1 - np.tan(angle_mean + 1e-10) * x2) \
                         / (np.tan(angle_mean + np.pi / 2 + 1e-10) - np.tan(angle_mean + 1e-10))
                    y3 = np.tan(angle_mean + 1e-10) * (x3 - x2) + y2
                    dist = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
                    if dist <= img_after.shape[1] / 40:
                        set_in_group = True
                        groups[j].append(i)
            if not set_in_group:
                groups.append([i])

        # for i in range(len(lin1)):
        #     print("{}% done".format(i / len(lin1) * 100))
        #     set_in_group = False
        #     x1 = (x_lines2[lin1[i]] + x_lines2[lin2[i]]) / 2
        #     y1 = (y_lines2[lin1[i]] + y_lines2[lin2[i]]) / 2
        #     angle = (angle_lines2[lin1[i]] + angle_lines2[lin2[i]]) / 2
        #     for j in range(len(groups)):
        #         angle_mean = np.mean(angle_lines2[groups[j]])
        #         x2 = np.mean(x_lines2[groups[j]])
        #         y2 = np.mean(y_lines2[groups[j]])
        #         if np.abs(angle - angle_mean) <= np.pi / 16:
        #             x3 = (y2 - y1 + np.tan(angle_mean + np.pi / 2 + 1e-10) * x1 - np.tan(angle_mean + 1e-10) * x2) \
        #                  / (np.tan(angle_mean + np.pi / 2 + 1e-10) - np.tan(angle_mean + 1e-10))
        #             y3 = np.tan(angle_mean + 1e-10) * (x3 - x2) + y2
        #             dist = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
        #             if dist <= img_after.shape[1] / 40:
        #                 set_in_group = True
        #                 groups[j].append(lin1[i])
        #                 groups[j].append(lin2[i])
        #     if not set_in_group:
        #         groups.append([lin1[i], lin2[i]])

        # for i in range(len(close_angles[0])):
        #     lin1_in_group = False
        #     lin2_in_group = False
        #     lin1_where = 0.1
        #     lin2_where = 0.1
        #     for j in range(len(groups)):
        #         if lin1[i] in groups[j]:
        #             lin1_in_group = True
        #             lin1_where = j
        #         if lin2[i] in groups[j]:
        #             lin2_in_group = True
        #             lin2_where = j
        #     if not lin1_in_group and not lin2_in_group:
        #         groups.append([lin1[i], lin2[i]])
        #     if lin1_in_group and not lin2_in_group:
        #         angle_mean = np.mean(angle_lines[groups[lin1_where]])
        #         if np.abs(angle_lines[lin2[i]]-angle_mean) <= np.pi/8:
        #             groups[lin1_where].append(lin2[i])
        #         else:
        #             placed_in_group = False
        #             for k in range(len(groups)):
        #                 angle_mean = np.mean(angle_lines[groups[k]])
        #
        #                 if np.abs(angle_lines[lin2[i]] - angle_mean) <= np.pi / 8:
        #                     groups[k].append(lin2[i])
        #                     placed_in_group = True
        #             if not placed_in_group:
        #                 groups.append([lin2[i]])
        #     if lin2_in_group and not lin1_in_group:
        #         angle_mean = np.mean(angle_lines[groups[lin2_where]])
        #         if np.abs(angle_lines[lin1[i]]-angle_mean) <= np.pi/8:
        #             groups[lin2_where].append(lin1[i])
        #         else:
        #             placed_in_group = False
        #             for k in range(len(groups)):
        #                 angle_mean = np.mean(angle_lines[groups[k]])
        #                 if np.abs(angle_lines[lin1[i]] - angle_mean) <= np.pi / 8:
        #                     groups[k].append(lin1[i])
        #                     placed_in_group = True
        #             if not placed_in_group:
        #                 groups.append([lin1[i]])
        #     if lin1_in_group and lin2_in_group and lin1_where != lin2_where:
        #         angle_mean_1 = np.mean(angle_lines[groups[lin1_where]])
        #         angle_mean_2 = np.mean(angle_lines[groups[lin2_where]])
        #         if np.abs(angle_mean_1-angle_mean_2) <= np.pi / 8:
        #             if lin1_where < lin2_where:
        #                 for j in range(len(groups[lin2_where])):
        #                     groups[lin1_where].append(groups[lin2_where][j])
        #                 groups.remove(groups[lin2_where])
        #             if lin1_where > lin2_where:
        #                 for j in range(len(groups[lin1_where])):
        #                     groups[lin2_where].append(groups[lin1_where][j])
        #                 groups.remove(groups[lin1_where])

        # print('successful!')
        for i in range(len(groups)):
            groups[i] = np.unique(groups[i])

        size_groups_combined = np.zeros((len(groups), len(groups)))
        for i in range(len(groups)):
            for j in np.arange(i + 1, len(groups), 1):
                if np.abs(np.mean(angle_lines2[groups[i]]) - np.mean(angle_lines2[groups[j]])) <= np.pi / 8:
                    angle_mean = (np.mean(angle_lines2[groups[i]]) * len(groups[i])
                                  + np.mean(angle_lines2[groups[j]]) * len(groups[j])) / \
                                 (len(groups[i]) + len(groups[j]))
                    x1 = np.mean(x_lines2[groups[i]])
                    y1 = np.mean(y_lines2[groups[i]])
                    x2 = np.mean(x_lines2[groups[j]])
                    y2 = np.mean(y_lines2[groups[j]])
                    x3 = (y2 - y1 + np.tan(angle_mean + np.pi / 2 + 1e-10) * x1 - np.tan(angle_mean + 1e-10) * x2) / \
                         (np.tan(angle_mean + np.pi / 2 + 1e-10) - np.tan(angle_mean + 1e-10))
                    y3 = np.tan(angle_mean + 1e-10) * (x3 - x2) + y2
                    dist = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
                    if (dist < img_after.shape[1] / 5) & (dist > img_after.shape[1] / 20):
                        size_groups_combined[i, j] = (len(groups[i]) + len(groups[j]))

        if np.max(size_groups_combined) > 0:
            biggest = np.where(size_groups_combined == np.max(size_groups_combined))
            after_grouping = []
            merge_big_groups = np.zeros((len(biggest[0])), dtype=bool)

            for k in range(len(biggest[0])):
                for l in np.arange(k+1, len(biggest[0]), 1):
                    angle_k = (np.mean(angle_lines2[groups[biggest[0][k]]])*len(groups[biggest[0][k]])
                               + np.mean(angle_lines2[groups[biggest[1][k]]])*len(groups[biggest[1][k]])) /\
                              (len(groups[biggest[0][k]])+len(groups[biggest[1][k]]))
                    angle_l = (np.mean(angle_lines2[groups[biggest[0][l]]])*len(groups[biggest[0][l]])
                               + np.mean(angle_lines2[groups[biggest[1][l]]])*len(groups[biggest[1][l]])) /\
                              (len(groups[biggest[0][l]])+len(groups[biggest[1][l]]))
                    if np.abs(angle_k - angle_l) <= np.pi/7:
                        merge_big_groups[k] = True
                        merge_big_groups[l] = True

            if np.sum(1*merge_big_groups) != 0:
                for k in np.where(merge_big_groups)[0]:
                    for i in groups[biggest[0][k]]:
                        after_grouping.append(i)
                    for j in groups[biggest[1][k]]:
                        after_grouping.append(j)

            else:  # if the biggest groups do not merge or there is only one group the first group is taken as result
                for i in groups[biggest[0][0]]:
                    after_grouping.append(i)
                for j in groups[biggest[1][0]]:
                    after_grouping.append(j)
            after_grouping = np.unique(after_grouping)
        else:
            after_grouping = np.array([])
        # biggest = 0
        # for i in range(len(groups)):
        #     if len(groups[i]) > len(groups[biggest]):
        #         biggest = i

        image1 = deepcopy(image)
        image2 = deepcopy(image)
        image3 = deepcopy(image)

        if len(after_grouping) > 0:
            print("{} lines after grouping".format(len(after_grouping)))

            for points in lines2[after_grouping]:
                # Extracted points nested in the list
                x1, y1, x2, y2 = points[0]
                # Draw the lines joining the points
                # On the original image
                cv2.line(image3, (x1, y1), (x2, y2), 255, 2)
                # Maintain a simples lookup list for points
                # lines_list.append([(x1, y1), (x2, y2)])
        else:
            print("0 lines after grouping")

        # for points in lines2[np.unique(close_angles)]:
        #     # Extracted points nested in the list
        #     x1, y1, x2, y2 = points[0]
        #     # Draw the lines joining the points
        #     # On the original image
        #     cv2.line(image2, (x1, y1), (x2, y2), 255, 2)
        #     # image2[int(y1/2+y2/2), int(x1/2+x2/2)] = 255
        #     # Maintain a simples lookup list for points
        #     # lines_list.append([(x1, y1), (x2, y2)])
        for points in lines:
            # Extracted points nested in the list
            x1, y1, x2, y2 = points[0]
            # Draw the lines joining the points
            # On the original image
            cv2.line(image, (x1, y1), (x2, y2), 255, 2)
            # Maintain a simples lookup list for points
            # lines_list.append([(x1, y1), (x2, y2)])
        for points in lines2:
            # Extracted points nested in the list
            x1, y1, x2, y2 = points[0]
            # Draw the lines joining the points
            # On the original image
            cv2.line(image1, (x1, y1), (x2, y2), 255, 2)
            # Maintain a simples lookup list for points
            # lines_list.append([(x1, y1), (x2, y2)])

        fig, ax = plt.subplots(ncols=2, nrows=3)
        ax[0, 0].imshow(img_after)
        ax[0, 0].set_title('img_after')
        ax[0, 1].imshow(edges)
        ax[0, 1].set_title('the edges')
        ax[1, 0].imshow(image2)
        ax[1, 0].set_title('after selection')
        ax[1, 1].imshow(image3)
        ax[1, 1].set_title('after grouping')
        ax[2, 1].imshow(image1)
        ax[2, 1].set_title('similar lines combined')
        ax[2, 0].imshow(image)
        ax[2, 0].set_title('all the lines')

        plt.show()
    else:
        print("No lines are detected")

    # # calculating the difference between the two images and creating a mask
    # mask = create_mask(img_before_blurred, img_after_blurred, threshold_mask, blur, squaring=False)
    # masked_img, extents = create_masked_img(img_after, mask, cropping)
    # mask2 = create_mask(img_before_blurred, img_after_blurred, threshold_mask, blur, squaring=squaring)
    # masked_img2, extents2 = create_masked_img(img_after, mask2, cropping)
    #
    # # setting a threshold for the image_after within the mask
    # binary_end_result1 = create_binary_end_image(masked_img, threshold_end, open_close=False)
    # binary_end_result2 = create_binary_end_image(masked_img2, threshold_end, open_close=True)
    #
    # print("binary image made")
    #
    # # here I start with trying to detect circles
    # key_points, yxr = detect_blobs(binary_end_result2, min_circ=0.6, min_area=8**2, plotting=False)
    #
    # print("circle detection complete")
    #
    # plot_end_results(img_before, img_after, img_before_blurred, img_after_blurred, mask, masked_img, masked_img2,
    #                  binary_end_result1, binary_end_result2, cropping, extents, extents2)

    del img_before, img_before_blurred, img_after, img_after_blurred, edges, image, \
        lines  # diff_angles, mask, mask2, masked_img, masked_img2, binary_end_result1, \
    # binary_end_result2, meta_before, meta_after
    gc.collect()

    print("Successful!\n")
