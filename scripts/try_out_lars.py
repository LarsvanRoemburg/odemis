from functions_lars import *
from odemis.dataio import tiff
import gc


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

for nnn in np.arange(0, 18, 1, dtype=int):  # range(len(data_paths_after)) OR np.arange(4, 9, 1, dtype=int)
    print("dataset nr. {}".format(nnn + 1))
    print(data_paths_before[nnn])

    # convert the image to a numpy array and set a threshold for outliers in the image / z-stack
    # for data before milling
    data_before_milling = tiff.read_data(data_paths_before[nnn])
    meta_before = data_before_milling[channel_before[nnn]].metadata

    img_before = z_projection_and_outlier_cutoff(nnn+1, data_before_milling, max_slices, channel_before[nnn], mode='max')
    del data_before_milling

    # for data after milling
    data_after_milling = tiff.read_data(data_paths_after[nnn])
    meta_after = data_after_milling[channel_after[nnn]].metadata

    img_after = z_projection_and_outlier_cutoff(nnn+1, data_after_milling, max_slices, channel_after[nnn], mode='max')
    del data_after_milling

    print("Pixel sizes are the same: {}".format(meta_before["Pixel size"][0] == meta_after["Pixel size"][0]))

    # rescaling one image to the other if necessary
    img_before, img_after = rescaling(img_before, img_after)

    # calculate the shift between the two images
    print("From convolution shift is:")
    img_after = overlay(img_before, img_after, max_shift=4)  # max_shift between 1 and inf

    diff = (np.array(meta_before['Centre position']) - np.array(meta_after['Centre position'])) / np.array(meta_after['Pixel size'])
    print("From metadata shift is: x: {}, y: {}, z: {}".format(diff[0], diff[1], diff[2]))

    # preprocessing steps of the images: blurring the image
    img_before, img_after, img_before_blurred, img_after_blurred = blur_and_norm(img_before, img_after, blur=blur)

    # detecting two parallel lines
    # the max difference in this projection is where there is milled
    # mid_milling_site = find_x_or_y_pos_milling_site(img_before_blurred, img_after_blurred)
    #
    # # here I try line detection
    # x_lines, y_lines, lines, edges = find_lines(img_after_blurred)
    # # lines = None
    # # Iterate over points
    #
    # # look at the lines and see if they have the same angle and rough distance between them
    # angle_lines = calculate_angles(lines)
    # x_lines2, y_lines2, lines2, angle_lines2 = combine_and_constraint_lines(x_lines, y_lines, lines, angle_lines,
    #                                                                         mid_milling_site,
    #                                                                         img_after_blurred.shape)
    #
    # groups = group_single_lines(x_lines2, y_lines2, lines2, angle_lines2, max_distance=img_after.shape[1] / 20)
    #
    # after_grouping = couple_groups_of_lines(groups, x_lines2, y_lines2, angle_lines2,
    #                                         max_dist=img_after.shape[1] / 5,
    #                                         min_dist=img_after.shape[1] / 20)

    # show_line_detection_steps(img_after, img_after_blurred, edges, lines, lines2, after_grouping)

    # calculating the difference between the two images and creating a mask
    mask = create_mask(img_before_blurred, img_after_blurred, squaring=False)
    masked_img, extents = create_masked_img(img_after, mask, cropping)
    mask2 = create_mask(img_before_blurred, img_after_blurred, squaring=True)
    masked_img2, extents2 = create_masked_img(img_after, mask2, cropping)

    # setting a threshold for the image_after within the mask
    binary_end_result1 = create_binary_end_image(masked_img, threshold_end, open_close=False)
    binary_end_result2 = create_binary_end_image(masked_img2, threshold_end, open_close=True)

    print("binary image made")

    # here I start with trying to detect circles
    # key_points, yxr = detect_blobs(binary_end_result2, min_circ=0.6, min_area=8**2, plotting=False)
    #
    # print("circle detection complete")

    plot_end_results(nnn+1, img_before, img_after, img_before_blurred, img_after_blurred, mask, masked_img, masked_img2,
                     binary_end_result1, binary_end_result2, cropping, extents, extents2)

    del img_before, img_before_blurred, img_after, img_after_blurred, \
        mask, mask2, masked_img, masked_img2, binary_end_result1, \
        binary_end_result2, meta_before, meta_after  # edges, lines,
    gc.collect()

    print("Successful!\n")
