from functions_lars import *
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

for nnn in np.arange(5, 18, 1, dtype=int):  # range(len(data_paths_after)) OR np.arange(4, 9, 1, dtype=int)
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
    img_after = overlay(img_before, img_after, max_shift=5)  # max_shift between 1 and inf

    # preprocessing steps of the images: blurring the image
    img_before, img_after, img_before_blurred, img_after_blurred = blur_and_norm(img_before, img_after, blur=blur)

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
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=1, minLineLength=img_after.shape[1] / 4,
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
            # Draw the lines joining the points
            # On the original image
            cv2.line(image, (x1, y1), (x2, y2), 255, 2)
            # Maintain a simples lookup list for points
            # lines_list.append([(x1, y1), (x2, y2)])
        plt.imshow(image)
    else:
        print("No lines are detected")

    # calculating the difference between the two images and creating a mask
    mask = create_mask(img_before_blurred, img_after_blurred, threshold_mask, blur, squaring=False)
    masked_img, extents = create_masked_img(img_after, mask, cropping)
    mask2 = create_mask(img_before_blurred, img_after_blurred, threshold_mask, blur, squaring=squaring)
    masked_img2, extents2 = create_masked_img(img_after, mask2, cropping)

    # setting a threshold for the image_after within the mask
    binary_end_result1 = create_binary_end_image(masked_img, threshold_end, open_close=False)
    binary_end_result2 = create_binary_end_image(masked_img2, threshold_end, open_close=True)

    print("binary image made")

    # here I start with trying to detect circles
    key_points, yxr = detect_blobs(binary_end_result2, min_circ=0.4, min_area=8**2, plotting=True)

    print("circle detection complete")

    plot_end_results(img_before, img_after, img_before_blurred, img_after_blurred, mask, masked_img, masked_img2,
                     binary_end_result1, binary_end_result2, cropping, extents, extents2)

    del img_before, img_before_blurred, img_after, img_after_blurred, mask, mask2, masked_img, masked_img2, \
        binary_end_result1, binary_end_result2, meta_before, meta_after
    gc.collect()

    print("Successful!\n")
