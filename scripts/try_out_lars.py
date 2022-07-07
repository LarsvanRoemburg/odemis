import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(module)s:%(lineno)d %(message)s")

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

# 2nd testdataset is with two milling sites
test_data_pico_before = [
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid1/Lam1_pre-milling_pico.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid1/Lam2_pre-milling_pico.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid1/Lam5_pre-milling_pico.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid1/Lam9_pre-milling_pico.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid2/lam5_pre-mill_picogreen.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid2/lam7_pre-mill_picogreen.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid2/lam8_pre-mill_picogreen.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid2/lam11_pre-mill_picogreen.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid3/lam10_pre-mill_pico.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/Yeast/lamella1_beforeMilling_EGFP.tif"
]

test_data_pico_after = [
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid1/Lam1-_postp3_pico.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid1/Lam2_postp3_picov2.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid1/Lam5_postp3_pico.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid1/Lam9_postp3_pico.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid2/lam5_post-mill_pico.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid2/lam7_post-mill_pico.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid2/lam8_post-mill_pico.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid2/lam11_post-mill_pico.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid3/lam10_post-mill_pico.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/Yeast/lamella1_Postp3_exp1.1sec-43-83_GFP.tif"
]

# 1st testdataset is with two milling sites
test_data_mito_before = [
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid1/Lam2_pre-milling_mitoT.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid1/Lam5_pre-milling_mitoT.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid1/Lam9_pre-milling_mitoT.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid2/lam5_pre-mill_mito.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid2/lam7_pre-mill_mito.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid2/lam8_pre-mill_mito.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid2/lam11_pre-mill_mito.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid3/lam10_pre-mill_mitoT.tiff"
]

test_data_mito_after = ["/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid1/Lam2_postp3_mitoTv2.tiff",
                        "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid1/Lam5_postp3_mitoT.tiff",
                        "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid1/Lam9_postp3_mitoT.tiff",
                        "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid2/lam5_post-mill_mito.tiff",
                        "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid2/lam7_post-mill_mito.tiff",
                        "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid2/lam8_post-mill_mito.tiff",
                        "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid2/lam11_post-mill_mito.tiff",
                        "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid3/lam10_post-mill_mito.tiff"
                        ]

# 2nd testdataset is with two milling sites
test_data_intra_before = [
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid1/Lam1_pre-milling_intracell.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid1/Lam2_pre-milling_intracell.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid1/Lam5_pre-milling_intracell.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid1/Lam9_pre-milling_intracell.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid2/lam5_pre-mill_intracell.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid2/lam7_pre-mill_intracell.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid2/lam8_pre-mill_intracell.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid2/lam11_pre-mill_intracell.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid3/lam10_pre-mill_intracell.tiff"
]

test_data_intra_after = [
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid1/Lam1-_postp3_intracell.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid1/Lam2_postp3_intracellv2.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid1/Lam5_postp3_intracell.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid1/Lam9_postp3_intracell.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid2/lam5_post-mill_intracell.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid2/lam7_post-mill_intracell.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid2/lam8_post-mill_intracell.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid2/lam11_post-mill_intracell.tiff",
    "/home/victoria/Documents/Lars/data/test data/ForDelmic/Cells/grid3/lam10_post-mill_intracell.tiff"
]

test_data_GFP_before = [
    "/home/victoria/Documents/Lars/data/test data/NEW_202205/Yeast_eGFP_Atg8/FOV1_new_checkpoint_00.tiff",
    "/home/victoria/Documents/Lars/data/test data/NEW_202205/Yeast_eGFP_Atg8/FOV3_new_checkpoint_00.tiff",
    "/home/victoria/Documents/Lars/data/test data/NEW_202205/Yeast_eGFP_Ede1/EA010_8_FOV3_checkpoint_00_1.tiff"]

test_data_GFP_after = [
    "/home/victoria/Documents/Lars/data/test data/NEW_202205/Yeast_eGFP_Atg8/FOV1_new_final_lamella.tiff",
    "/home/victoria/Documents/Lars/data/test data/NEW_202205/Yeast_eGFP_Atg8/FOV3_new_final_lamella.tiff",
    "/home/victoria/Documents/Lars/data/test data/NEW_202205/Yeast_eGFP_Ede1/EA010_8_FOV3_final.tiff"]

yeast_not_mammalian = np.ones(18, dtype=bool)
yeast_not_mammalian[8:12] = False
signal_in_data = ['No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Maybe', 'Yes', 'Yes',
                  'Yes', 'Yes', 'Yes', 'No', 'No']

if __name__ == '__main__':
    ll = len(data_paths_before)
    channel_before = np.zeros(ll, dtype=int)
    # channel_before[7] = 1
    channel_after = np.zeros(ll, dtype=int)

    threshold_mask = 0.3
    threshold_end = 0.3
    blur = 25
    max_slices = 30
    cropping = True  # if true, the last images will be cropped to only the mask

    signal_in_yes = []
    signal_in_no = []
    signal_in_maybe = []
    answers = []
    signal_in_yes2 = []
    signal_in_no2 = []
    signal_in_maybe2 = []
    answers2 = []

    for nnn in np.arange(17, len(data_paths_before), 1,
                         dtype=int):  # range(len(data_paths_after)) OR np.arange(4, 9, 1, dtype=int)
        print("dataset nr. {}".format(nnn + 1))
        print(data_paths_before[nnn])
        print(data_paths_after[nnn])

        logging.info('starting ROI detection workflow')
        # convert the image to a numpy array and set a threshold for outliers in the image / z-stack
        # for data before milling
        img_before, img_after, meta_before, meta_after = get_image(data_paths_before[nnn],
                                                                   data_paths_after[nnn],
                                                                   channel_before[nnn], channel_after[nnn],
                                                                   mode='in_focus', proj_mode='max')


        logging.info("get_image() done")
        img_before, img_after, img_before_blurred, img_after_blurred = pre_processing_data(img_before, img_after)

        logging.info("preprocessing done")

        # best_template, best_angle, best_y, best_x = match_template_to_image(img_before, img_after)
        # template_mask = get_template_mask(img_after, best_template, best_angle, best_y, best_x)
        # logging.info("template matching done")

        mask_diff, mask_combined, combined = get_mask(img_before_blurred, img_after_blurred, plotting_lines=False)
        logging.info("get_mask() done")

        masked_img, extents, binary_end, binary_end_without, key_points, yxr, try_again = analyze_data(img_after,
                                                                                                       mask_combined,
                                                                                                       plotting=False)
        masked_img2, extents2, binary_end2, binary_end_without2, key_points2, yxr2, try_again2 = analyze_data(img_after,
                                                                                                              mask_diff)
        logging.info("analyzing the data done")

        binary_img, b_img_without = from_blobs_to_binary(yxr, masked_img.shape, mask_combined)

        signal, answer = from_binary_to_answer(binary_end_without, masked_img, try_again)

        signal2, answer2 = from_binary_to_answer(b_img_without, masked_img, try_again)
        # 2 works better!
        logging.info('final answer done')
        if 'Yes' in signal_in_data[nnn]:
            signal_in_yes.append(signal)
            signal_in_yes2.append(signal2)
        elif 'No' in signal_in_data[nnn]:
            signal_in_no.append(signal)
            signal_in_no2.append(signal2)
        elif 'Maybe' in signal_in_data[nnn]:
            signal_in_maybe.append(signal)
            signal_in_maybe2.append(signal2)
        answers.append(answer)
        answers2.append(answer2)

        print(f"the signal intensity/m2 is: {'%.3g' % signal2}")
        print(f"the predicted answer is: {answer2}")
        # print(f"the correct answer is: {signal_in_data[nnn]}")

        plot_end_results(img_before, img_after, img_before_blurred, img_after_blurred, mask_diff, masked_img,
                         masked_img2, binary_end_without, binary_end_without2, cropping, extents, extents2)
        logging.info('plotting results done')
        del img_before, img_after, img_before_blurred, img_after_blurred, \
            mask_combined, mask_diff, masked_img, masked_img2, binary_end, \
            binary_end2, meta_before, meta_after
        gc.collect()

        logging.info("ROI detection workflow was successful!\n")

    # plt.show()
    res = 0
    res2 = 0
    for i in range(len(answers)):
        res += 1 * (answers[i] == signal_in_data[i])
        res2 += 1 * (answers2[i] == signal_in_data[i])
    print(f"{res} out of {len(answers)} are correctly predicted with binary thresholding")
    print(f"{res2} out of {len(answers)} are correctly predicted with blob detection")
    print("Ending data set")
