from functions_lars import *


def pre_processing_data(img_before, img_after, blur=25, max_shift=4):
    # rescaling one image to the other if necessary
    img_before, img_after, magni = rescaling(img_before, img_after)

    # calculate the shift between the two images
    img_after, shift = overlay(img_before, img_after, max_shift=max_shift)  # max_shift between 1 and inf

    # preprocessing steps of the images: blurring the image
    img_before, img_before_blurred = blur_and_norm(img_before, blur=blur)
    img_after, img_after_blurred = blur_and_norm(img_after, blur=blur)

    return img_before, img_after, img_before_blurred, img_after_blurred


def get_mask(img_before_blurred, img_after_blurred, max_dist_s=1 / 30, min_dist_c=1 / 25, max_dist_c=1 / 5,
             max_angle_diff=np.pi / 8, plotting_lines=False):
    # detecting two parallel lines
    # the max difference in this projection is where there is milled
    mid_milling_site = find_x_or_y_pos_milling_site(img_before_blurred, img_after_blurred)

    # here I try line detection
    x_lines, y_lines, lines, edges = find_lines(img_after_blurred)
    # lines = np.array([])
    # apparently for mammalian cell samples, setting the hysteresis threshold to 0 and 0 is better

    # look at the lines and see if they have the same angle and rough distance between them
    angle_lines = calculate_angles(lines)
    x_lines2, y_lines2, lines2, angle_lines2 = combine_and_constraint_lines(x_lines, y_lines, lines, angle_lines,
                                                                            mid_milling_site,
                                                                            img_after_blurred.shape,
                                                                            max_dist=max_dist_s)

    groups = group_single_lines(x_lines2, y_lines2, lines2, angle_lines2,
                                max_distance=img_after_blurred.shape[1] * max_dist_s,
                                max_angle_diff=max_angle_diff)

    after_grouping = couple_groups_of_lines(groups, x_lines2, y_lines2, angle_lines2, mid_milling_site,
                                            max_dist=img_after_blurred.shape[1] * max_dist_c,
                                            min_dist=img_after_blurred.shape[1] * min_dist_c,
                                            max_angle_diff=max_angle_diff)

    if plotting_lines:
        show_line_detection_steps(img_after_blurred, edges, lines, lines2, after_grouping)

    mask_lines = create_line_mask(after_grouping, x_lines2, y_lines2, lines2, angle_lines2, img_after_blurred.shape,
                                  all_groups=False, inv_max_dist=int(1 / max_dist_s), max_angle=max_angle_diff)
    mask_lines_all = create_line_mask(after_grouping, x_lines2, y_lines2, lines2, angle_lines2, img_after_blurred.shape,
                                      all_groups=True, inv_max_dist=int(1 / max_dist_s), max_angle=max_angle_diff)

    mask = create_diff_mask(img_before_blurred, img_after_blurred, squaring=False)
    mask_combined, combined = combine_masks(mask, mask_lines, mask_lines_all, squaring=True)
    mask = create_diff_mask(img_before_blurred, img_after_blurred, squaring=True)

    return mask, mask_combined, combined


def analyze_data(img_after, mask_combined, cropping, threshold_end=0.25, open_close=True, rid_bg=True, b1=1, b2=4,
                 b3=6, b4=10, min_circ=0.6, max_circ=1.01, min_area=0, max_area=np.inf, min_in=0, max_in=1.01,
                 min_con=0, max_con=1.01, plotting=False):
    """
    j
    j
    k
    """

    masked_img, extents = create_masked_img(img_after, mask_combined, cropping)
    binary_end, binary_end_without = create_binary_end_image(mask_combined, masked_img, threshold=threshold_end,
                                                             open_close=open_close, rid_of_back_signal=rid_bg, b1=b1,
                                                             b2=b2, b3=b3, b4=b4)
    key_points, yxr = detect_blobs(binary_end_without, min_circ=min_circ, max_circ=max_circ, min_area=min_area,
                                   max_area=max_area, min_in=min_in, max_in=max_in, min_con=min_con, max_con=max_con,
                                   plotting=plotting)

    return masked_img, extents, binary_end, binary_end_without, key_points, yxr
