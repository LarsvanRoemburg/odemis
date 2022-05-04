from functions_lars import *
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

data_paths_true_masks = ["/home/victoria/Documents/Lars/data/1/true_mask_FOV2_GFP_cp04.tif",
                         "/home/victoria/Documents/Lars/data/2/true_mask_FOV1_checkpoint_01.tiff",
                         "/home/victoria/Documents/Lars/data/3/true_mask_FOV4_checkpoint_01.tiff",
                         "/home/victoria/Documents/Lars/data/4/METEOR_images/true_mask_FOV3_checkpoint_01.tif",
                         "/home/victoria/Documents/Lars/data/6/true_mask_20201002_FOV2_checkpoint_005_stack_001.tiff",
                         "/home/victoria/Documents/Lars/data/7/FOV3_Meteor_stacks/Lamella_slices_tif"
                         "/true_mask_EA010_8_FOV3_final-19.tif",
                         "/home/victoria/Documents/Lars/data/meteor1/FOV3_Meteor_stacks/"
                         "true_mask_EA010_8_FOV3_final.tiff",
                         "/home/victoria/Documents/Lars/data/XA Yeast-20220215T093027Z-001/XA "
                         "Yeast/true_mask_20200918-zstack_400nm.tiff",
                         "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Mammalian_cells"
                         "/true_mask_FOV7_after_GFP.tif",
                         "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Mammalian_cells"
                         "/true_mask_FOV9_after_GFP.tif",
                         "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Mammalian_cells"
                         "/true_mask_FOV11_after_GFP.tif",
                         "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Mammalian_cells"
                         "/true_mask_FOV12_after_GFP.tif",
                         "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Yeast_eGFP-Ede1"
                         "/true_mask_G2_FOV1_checkpoint03.tif",
                         "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Yeast_eGFP-Ede1"
                         "/true_mask_G2_FOV2_checkpoint04.tif",
                         "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Yeast_eGFP-Ede1"
                         "/true_mask_G2_FOV3_checkpoint04.tif",
                         "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Yeast_eGFP-Ede1"
                         "/true_mask_G2_FOV4_checkpoint04.tif",
                         "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/negative_examples/yeast/"
                         "true_mask_FOV3_new_final_lamella.tiff",
                         "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/negative_examples/yeast/"
                         "true_mask_FOV6_checkpoint02_ch00.tiff"
                         ]



if __name__ == '__main__':
    ll = len(data_paths_before)
    channel_before = np.zeros(ll, dtype=int)
    channel_before[7] = 1
    channel_after = np.zeros(ll, dtype=int)

    threshold_mask = 0.3
    threshold_end = 0.25
    blur = 25
    max_slices = 30
    cropping = True  # if true, the last images will be cropped to only the mask

    for nnn in np.arange(2, 18, 1, dtype=int):  # range(len(data_paths_after)) OR np.arange(4, 9, 1, dtype=int)
        print("dataset nr. {}".format(nnn + 1))
        print(data_paths_before[nnn])
        print(data_paths_after[nnn])

        # convert the image to a numpy array and set a threshold for outliers in the image / z-stack
        # for data before milling
        img_before, img_after, meta_before, meta_after = get_image(data_paths_before[nnn], data_paths_after[nnn],
                                                                   channel_before[nnn], channel_after[nnn],
                                                                   mode='in_focus', proj_mode='max')

        print("Pixel sizes are the same: {}".format(meta_before["Pixel size"][0] == meta_after["Pixel size"][0]))

        # rescaling one image to the other if necessary
        img_before, img_after, magni = rescaling(img_before, img_after)

        # calculate the shift between the two images
        img_after, shift = overlay(img_before, img_after, max_shift=4)  # max_shift between 1 and inf

        # preprocessing steps of the images: blurring the image
        img_before, img_after, img_before_blurred, img_after_blurred = blur_and_norm(img_before, img_after, blur=blur)

        # detecting two parallel lines
        # the max difference in this projection is where there is milled
        mid_milling_site = find_x_or_y_pos_milling_site(img_before_blurred, img_after_blurred)
        mid_milling_site_y = find_x_or_y_pos_milling_site(img_before_blurred, img_after_blurred, ax='y')
        print(f"mid_milling_x = {mid_milling_site}\nmid_milling_y = {mid_milling_site_y}")
        # plt.imshow(img_after_blurred)
        # plt.show()

        # here I try line detection
        x_lines, y_lines, lines, edges = find_lines(img_after_blurred)
        # apparently for mammalian cell samples, setting the hysteresis threshold to 0 and 0 is better

        # look at the lines and see if they have the same angle and rough distance between them
        angle_lines = calculate_angles(lines)
        x_lines2, y_lines2, lines2, angle_lines2 = combine_and_constraint_lines(x_lines, y_lines, lines, angle_lines,
                                                                                mid_milling_site,
                                                                                img_after_blurred.shape, max_dist=1/30)

        groups = group_single_lines(x_lines2, y_lines2, lines2, angle_lines2, max_distance=img_after.shape[1] / 30)

        after_grouping = couple_groups_of_lines(groups, x_lines2, y_lines2, angle_lines2, mid_milling_site,
                                                max_dist=img_after.shape[1] / 5,
                                                min_dist=img_after.shape[1] / 25)

        show_line_detection_steps(img_after, img_after_blurred, edges, lines, lines2, after_grouping)

        mask_lines = create_line_mask(after_grouping, x_lines2, y_lines2, lines2, angle_lines2, img_after.shape,
                                      all_groups=False)
        mask_lines_all = create_line_mask(after_grouping, x_lines2, y_lines2, lines2, angle_lines2, img_after.shape,
                                          all_groups=True)

        # calculating the difference between the two images and creating a mask
        mask = create_diff_mask(img_before_blurred, img_after_blurred, squaring=False)

        mask2 = create_diff_mask(img_before_blurred, img_after_blurred, squaring=True)
        masked_img2, extents2 = create_masked_img(img_after, mask2, cropping)

        mask_combined, combined = combine_masks(mask, mask_lines, mask_lines_all, squaring=False)

        masked_img, extents = create_masked_img(img_after, mask_combined, cropping)

        # fig3, ax3 = plt.subplots(ncols=4)
        # ax3[0].imshow(mask)
        # ax3[0].set_title("Difference mask")
        # ax3[1].imshow(mask_lines)
        # ax3[1].set_title("Line mask")
        # ax3[2].imshow((5.0*mask_lines + 5.0*mask))
        # ax3[2].set_title("Overlap")
        # ax3[3].imshow(mask_combined)
        # ax3[3].set_title("Masks combined")

        # setting a threshold for the image_after within the mask
        binary_end_1b, binary_end_result1 = create_binary_end_image(mask_combined, masked_img, threshold_end,
                                                                    open_close=True)
        binary_end_2b, binary_end_result2 = create_binary_end_image(mask2, masked_img2, threshold_end, open_close=True,
                                                                    rid_of_back_signal=True)

        # fig, ax = plt.subplots(ncols=2)
        # ax[0].imshow(binary_end_result1)
        # ax[1].imshow(binary_end_1b)
        # plt.show()
        # print("binary image made")

        # here I start with trying to detect circles
        key_points, yxr = detect_blobs(binary_end_result2, min_circ=0.6, min_area=7**2/2, plotting=False)

        print("circle detection complete")

        plot_end_results(img_before, img_after, img_before_blurred, img_after_blurred, mask, masked_img,
                         masked_img2, binary_end_result1, binary_end_result2, cropping, extents, extents2)

        del img_before, img_after, img_before_blurred, img_after_blurred, \
            mask, mask2, masked_img, masked_img2, binary_end_result1, \
            binary_end_result2, meta_before, meta_after, edges, lines
        gc.collect()

        print("Successful!\n")
