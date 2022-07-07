from functions_lars import *


def pre_processing_data(img_before, img_after, blur=25, max_shift=4):
    """
    The pre-processing functions combined into one. First, the images are rescaled to each other if necessary. Second,
    the images are shifted to each other, so they overlap nicely. And third, the images are blurred and normalized for
    further processing. See the individual functions docstrings for more information.

    Parameters:
        img_before (ndarray):           The image before milling from get_image().
        img_after (ndarray):            The image after milling from get_image().
        blur (int):                     The amount of blur applied to the images, the bigger the number, the more the
                                        blur. Needs to be an integer.
       max_shift (int):                 It is assumed that the images are roughly at the same position,
                                        so no large shifts are needed. On all sides of the convolution image,
                                        1/max_shift of the image is set to be zero. So with 1/8 you have a max shift
                                        of 3/8 in x and y direction (-3/8*img_shape, 3/8*img_shape). If you don't want
                                        a constraint on the shift, set max_shift to <=1.

    Returns:
        img_before (ndarray):           The normalized image before milling without blurring.
        img_after (ndarray):            The normalized image after milling without blurring
        img_before_blurred (ndarray):   The normalized image before milling with blurring.
        img_after_blurred (ndarray):    The normalized image after milling with blurring.
    """

    # rescaling one image to the other if necessary
    img_before, img_after, magni = rescaling(img_before, img_after)

    # calculate the shift between the two images
    img_before, img_after, shift = overlay(img_before, img_after, max_shift=max_shift)  # max_shift between 1 and inf

    # preprocessing steps of the images: blurring the image
    img_before, img_before_blurred = blur_and_norm(img_before, blur=blur)
    img_after, img_after_blurred = blur_and_norm(img_after, blur=blur)

    return img_before, img_after, img_before_blurred, img_after_blurred


def get_mask(img_before_blurred, img_after_blurred, max_dist_s=1 / 30, min_dist_c=1 / 25, max_dist_c=1 / 5,
             max_angle_diff=np.pi / 8, low_thres=0.1, high_thres=1.5, plotting_lines=False):
    """
    Here all the functions working towards a mask of the milling site are combined. First, the line detection functions
    are used for creating a mask with the found lines. Second, a mask is created with the intensity differences in the
    images before and after milling. Third, the two masks are combined into one if possible and returned as output.

    Parameters:
        img_before_blurred (ndarray):   The normalized image before milling with blurring.
        img_after_blurred (ndarray):    The normalized image after milling with blurring.
        max_dist_s (float):             The maximum distance single lines can have to be combined into one line.
                                        This number is in fraction of the image size.
        min_dist_c (float):             The minimum distance for combining two groups of lines into one milling site
                                        group. This number is in fraction of the image size.
        max_dist_c (float):             The maximum distance for combining two groups of lines into one milling site
                                        group. This number is in fraction of the image size.
        max_angle_diff (float):         The maximum angle difference lines or groups of lines can have to be
                                        combined or grouped in combine_and_constraint_lines(), group_single_lines() and
                                        couple_groups_of_lines(). This value is in the unit radians.
        low_thres (float):              The lower threshold of the hysteresis in the hough line detection.
                                        (see find_lines() for more).
        high_thres (float):              The higher threshold of the hysteresis in the hough line detection.
                                        (see find_lines() for more).
        plotting_lines (bool):          A boolean to indicate if we want to plot the found lines or not before
                                        continuing to the next step.

    Returns:
        mask (ndarray):                 The mask from the intensity differences.
        mask_combined (ndarray):        The mask combined from the intensity differences and the line detection.
                                        If they were not able to be combined, it is the mask from the intensity
                                        differences.
        combined (bool):                A boolean stating if the two masks were able to combine.
    """

    # detecting two parallel lines
    # the max difference in this projection is where there is milled
    mid_milling_site = find_x_or_y_pos_milling_site(img_before_blurred, img_after_blurred)

    # here I try line detection
    x_lines, y_lines, lines, edges = find_lines(img_after_blurred, low_thres_edges=low_thres,
                                                high_thres_edges=high_thres)
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


def analyze_data(img_after, mask_combined, cropping=True, threshold_end=0.25, open_close=True, rid_bg=True, min_circ=0.6,
                 max_circ=1.01, min_area=0, max_area=np.inf, plotting=False):
    """
    Here the last analytical functions are combined to get an answer to the question if there is any signal left in the
    milling site after milling. First, the image after milling within the milling site is created (cropped or not
    cropped). Second, a simple threshold with multiple binary operations are performed to get a binary output image.
    Third, a blob detection is used on the binary image to find specific features.

    Parameters:
          img_after (ndarray):          The image after milling.
          mask_combined (ndarray):      The mask of the milling site.
          cropping (bool):              A boolean stating if the image needs to be cropped around the mask or not.
          threshold_end (float):        The value at which a threshold is set at the image after milling.
          open_close (bool):            A boolean stating if some closing and opening binary operations are used at the
                                        binary image after thresholding.
          rid_bg (bool):                A boolean stating if you want to get rid of signal at the boundaries of the
                                        mask.
          min_circ (float):             The minimum circularity blobs need to have to be detected in detect_blobs().
                                        It needs to be in the range from 0 to 1.01.
          max_circ (float):             The maximum circularity blobs need to have to be detected in detect_blobs().
                                        It needs to be in the range from 0 to 1.01.
          min_area (float):             The minimum area blobs need to have to be detected in detect_blobs().
          max_area (float):             The maximum area blobs need to have to be detected in detect_blobs().
          plotting (bool):              A boolean stating if the blob detection output needs to be shown with
                                        matplotlib or not.

    Returns:
        masked_img (ndarray):           The image after milling only at the mask.
        extents (tuple):                The (x_min, x_max, y_max, y_min) of the mask in the full image to which the
                                        image is cropped to.
        binary_end (ndarray):           A binary image which shows the thought to be signal in the masked image.
        binary_end_without (ndarray):   A binary image which shows the thought to be signal in the masked image without
                                        the signal on the boundary of the mask.
        key_points (tuple):             A tuple of opencv KeyPoints which is the raw output of the opencv
                                        SimpleBlobDetector in the detect_blobs() function.
        yxr (ndarray):                  The y-, x- and r-values from the key_points in a numpy array with shape (#n, 3)
                                        with #n being the number of blobs. The y and x are the y and x center positions
                                        of the blob and the r is the radius of the blob if it was a circle (so r is not
                                        representable in blobs that are far away from the shape of a circle).
    """

    masked_img, extents = create_masked_img(img_after, mask_combined, cropping)
    binary_end, binary_end_without = create_binary_end_image(mask_combined, masked_img, threshold=threshold_end,
                                                             open_close=open_close, rid_of_back_signal=rid_bg)
    key_points, yxr = detect_blobs(gaussian_filter(masked_img, sigma=1), min_circ=0.3, max_circ=1.01, min_area=10,
                                   max_area=50**2, plotting=plotting)
    binary_img, b_img_without = from_blobs_to_binary(yxr, masked_img.shape, mask_combined)
    try_again = False
    if np.sum(b_img_without) == 0:
        key_points, yxr = detect_blobs(gaussian_filter(masked_img, sigma=1), min_thres=0.15, max_thres=0.3,
                                       min_circ=0.3, max_circ=1.01, min_area=10, max_area=50 ** 2, plotting=plotting)
        try_again = True
        logging.info("No blobs were found with the standard threshold, a lower threshold is set.")

    return masked_img, extents, binary_end, binary_end_without, key_points, yxr, try_again

# TODO: in all binary operations it needs to become pixel resolution independent
