from meta_functions_lars import *
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
    threshold_end = 0.3
    blur = 25
    max_slices = 30
    cropping = True  # if true, the last images will be cropped to only the mask

    for nnn in np.arange(2, 3, 1, dtype=int):  # range(len(data_paths_after)) OR np.arange(4, 9, 1, dtype=int)
        print("dataset nr. {}".format(nnn + 1))
        print(data_paths_before[nnn])
        print(data_paths_after[nnn])

        # convert the image to a numpy array and set a threshold for outliers in the image / z-stack
        # for data before milling
        img_before, img_after, meta_before, meta_after = get_image(data_paths_before[nnn], data_paths_after[nnn],
                                                                   channel_before[nnn], channel_after[nnn],
                                                                   mode='in_focus', proj_mode='max')

        img_before, img_after, img_before_blurred, img_after_blurred = pre_processing_data(img_before, img_after)
        # img_after = downsize_image(gaussian_filter(img_after, sigma=5), 4)

        match_template_to_image(img_before, img_after)
        # mask_diff, mask_combined, combined = get_mask(img_before_blurred, img_after_blurred)
        #
        # masked_img, extents, binary_end, binary_end_without, key_points, yxr = analyze_data(img_after, mask_combined)
        # masked_img2, extents2, binary_end2, binary_end_without2, key_points2, yxr2 = analyze_data(img_after, mask_diff)
        #
        # plot_end_results(img_before, img_after, img_before_blurred, img_after_blurred, mask_diff, masked_img,
        #                  masked_img2, binary_end_without, binary_end_without2, cropping, extents, extents2)

        # del img_before, img_after, img_before_blurred, img_after_blurred, \
            # mask_combined, mask_diff, masked_img, masked_img2, binary_end, \
            # binary_end2, meta_before, meta_after
        gc.collect()

        print("Successful!\n")
    plt.show()
