import gc
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import binary_opening, binary_closing, binary_erosion, binary_dilation
from skimage.draw import circle_perimeter
from odemis.dataio import tiff
import cv2


def find_lines_bas(img_after_blurred, blur=10, low_thres_edges=0.1, high_thres_edges=2.5, angle_res=np.pi / 180,
                   line_thres=1, min_len_lines=1 / 45, max_line_gap=1 / 40):
    """
    This function finds all the lines in an image with the canny (edge detection) function of skimage and the line
    detection function of opencv. You can set constraints to this line detection with the parameters of this function.
    If no lines are found, None is returned for everything.

    For more information on the canny and HoughLines functions:
    https://scikit-image.org/docs/stable/api/skimage.feature.html?highlight=module%20feature#skimage.feature.canny
    https://docs.opencv.org/3.4/d3/de6/tutorial_js_houghlines.html

    Parameters:
        img_after_blurred (ndarray):    The image on which line detection is used.
        blur (int):                     The (extra) blurring done on the image for edge detection.
        low_thres_edges (float):        The lower hysteresis threshold passed into the canny edge detection.
        high_thres_edges (float):       The upper hysteresis threshold passed into the canny edge detection.
        angle_res (float):              The angular resolution with which lines are detected in the opencv function.
        line_thres (int):               The quality threshold for lines to be detected.
        min_len_lines (float):          The minimal length of the lines detected as a fraction of the image size.
        max_line_gap (float):           The maximal gap within the detected lines as a fraction of the image size.

    Returns:
        x_lines ():                     The x-center of the lines detected.
        y_lines ():                     The y-center of the lines detected.
        lines (ndarray):                The lines detected in the format (#lines, 1, 4).
                                        The last index gives the positions of the ends of the lines:
                                        (0: x of end 1, 1: y of end 1, 2: x of end 2, 3: y of end 2).
        edges (ndarray):                The output image of the edge detection.
    """

    image = np.array(img_after_blurred, dtype=np.uint8)
    # for visualization purposes the image is inverted (not necessary)
    # image = cv2.bitwise_not(image)

    shape_1 = image.shape[1]
    min_len_lines = shape_1 * min_len_lines
    max_line_gap = shape_1 * max_line_gap

    # finding the edges via the canny function and the lines with the hough lines function
    image = gaussian_filter(image, sigma=blur)
    edges = cv2.Canny(image, threshold1=high_thres_edges, threshold2=low_thres_edges)
    lines = cv2.HoughLinesP(np.array(edges, dtype=np.uint8), 1, angle_res, threshold=line_thres,
                            minLineLength=min_len_lines, maxLineGap=max_line_gap)

    if lines is not None:
        print("{} lines detected.".format(len(lines)))
        # here we extract the middle points of the lines
        x_lines = (lines.T[2] / 2 + lines.T[0] / 2).T.reshape((len(lines),))
        y_lines = (lines.T[3] / 2 + lines.T[1] / 2).T.reshape((len(lines),))
        return x_lines, y_lines, lines, edges
    else:
        print("No lines are detected at all!")
        return None, None, None, None


def calculate_angles(lines):
    """
    Here the angles of the lines are calculated with a simple arctan. It will have the range of 0 to pi.

    Parameters:
        lines (ndarray):        The end positions of the lines as outputted from find_lines,
                                in the format (#lines, 1, 4) with for the last part:
                                (0: x of end 1, 1: y of end 1, 2: x of end 2, 3: y of end 2).

    Returns:
        angle_lines (ndarray):  The angles of the lines in a single 1D-array.
    """

    if lines is None:
        return None

    angle_lines = np.zeros(lines.shape[0])

    for i in range(lines.shape[0]):
        if lines[i][0][2] != lines[i][0][0]:
            angle_lines[i] = np.arctan((lines[i][0][3] - lines[i][0][1]) / (lines[i][0][2] - lines[i][0][0]))

            if angle_lines[i] < 0:
                angle_lines[i] = angle_lines[i] + np.pi

        else:
            angle_lines[i] = np.pi / 2

    return angle_lines


def combine_and_constraint_lines(x_lines, y_lines, lines, angle_lines, mid_milling_site, img_shape,
                                 x_width_constraint=1 / 4, y_width_constraint=3 / 4, angle_constraint=np.pi / 7,
                                 max_dist=1 / 25, max_diff_angle=np.pi / 10):
    """
    Here lines which are roughly at the same spot and angle are combined in one line.
    Next to that, also some constraints can be given for which lines to pick out for further processing.
    Lines at an angle of around 0 or pi are discarded because no solution on the boundary here has been implemented.

    Parameters:
        x_lines (ndarray):              The x-centers of the lines.
        y_lines (ndarray):              The y-centers of the lines.
        lines (ndarray):                The end positions of the lines as outputted from find_lines.
        angle_lines (ndarray):          The angles of the lines as outputted from calculate_angles.
        mid_milling_site (int):         The x center position of the milling site.
        img_shape (tuple):              The shape of the image.
        x_width_constraint (float):     The range of the x centers lines should have around the milling site for further
                                        processing as a fraction of the image shape.
        y_width_constraint (float):     The range of the y centers lines should have around the middle of the image
                                        for further processing as a fraction of the image shape.
        angle_constraint (float):       The angle of the lines have to be bigger than the angle_constraint and smaller
                                        than pi - angle_constraint.
        max_dist (float):               The maximum distance lines can have between their center positions to merge.
        max_diff_angle (float):         The maximum difference in angles lines can have to merge.

    Returns:
        x_lines (ndarray):              The x-center of the lines after merging and selection.
        y_lines (ndarray):              The y-center of the lines after merging and selection.
        lines (ndarray):                The end positions of the lines after merging and selection.
        angle_lines (ndarray):          The angles of the lines after merging and selection.
    """

    if lines is None:
        return None, None, None, None
    x_constraint = img_shape[1] * x_width_constraint / 2
    # in which region you leave the lines exist : mid_mil_site +- x_constraint
    y_constraint = img_shape[0] * y_width_constraint / 2
    # in which region you leave the lines exist: y_const < y < y_shape - y_const
    max_dist_for_merge = img_shape[0] * max_dist  # the max distance lines can have for merging

    print("mid_milling_site = {}".format(mid_milling_site))
    print("lower x boundary = {}".format(mid_milling_site - x_constraint))
    print("upper x boundary = {}".format(mid_milling_site + x_constraint))

    # for each line there exists a spot on num_lines_present, if set to 0 it is discarded
    num_lines_present = np.ones(lines.shape[0])
    for i in range(lines.shape[0]):
        x1 = x_lines[i]
        y1 = y_lines[i]

        # for each line, apply the constraints and if it does not pass, discard it
        if (angle_lines[i] <= angle_constraint) | (angle_lines[i] >= (np.pi - angle_constraint)) | \
                (x1 < (mid_milling_site - x_constraint)) | \
                (x1 > (mid_milling_site + x_constraint)) | \
                (y1 > (img_shape[0] / 2 + y_constraint)) | (y1 < (img_shape[0] / 2 - y_constraint)):
            num_lines_present[i] = 0

        else:
            # If passed, look at all the other lines (that are still present) and calculate the distance between them
            # and the difference in angle. If two lines are suited for merging, they are merged and put together at
            # one spot in num_lines_present.
            for j in range(lines.shape[0]):
                if (i != j) & (num_lines_present[i] >= 1) & (num_lines_present[j] >= 1):
                    # calculating the distance and angle difference
                    dist = np.sqrt((x_lines[i] - x_lines[j]) ** 2 + (y_lines[i] - y_lines[j]) ** 2)
                    diff_angle = np.abs(angle_lines[i] - angle_lines[j])

                    # checking merging conditions
                    if (dist <= max_dist_for_merge) & (diff_angle <= max_diff_angle):
                        # put the lines together at the smallest index
                        if i < j:
                            # Calculating the average end points, middle points and angles of the two weighted lines
                            # and put it directly in the lines array. With weighted I mean how many lines are already
                            # put at the two spots which we will merge.
                            lines[i][0][0] = (lines[i][0][0] * num_lines_present[i] + lines[j][0][0] *
                                              num_lines_present[j]) / (num_lines_present[i] + num_lines_present[j])
                            lines[i][0][1] = (lines[i][0][1] * num_lines_present[i] + lines[j][0][1] *
                                              num_lines_present[j]) / (num_lines_present[i] + num_lines_present[j])
                            lines[i][0][2] = (lines[i][0][2] * num_lines_present[i] + lines[j][0][2] *
                                              num_lines_present[j]) / (num_lines_present[i] + num_lines_present[j])
                            lines[i][0][3] = (lines[i][0][3] * num_lines_present[i] + lines[j][0][3] *
                                              num_lines_present[j]) / (num_lines_present[i] + num_lines_present[j])
                            angle_lines[i] = (angle_lines[i] * num_lines_present[i]
                                              + angle_lines[j] * num_lines_present[j]) / \
                                             (num_lines_present[i] + num_lines_present[j])
                            x_lines[i] = (x_lines[i] * num_lines_present[i] + x_lines[j] * num_lines_present[
                                j]) / (num_lines_present[i] + num_lines_present[j])
                            y_lines[i] = (y_lines[i] * num_lines_present[i] + y_lines[j] * num_lines_present[
                                j]) / (num_lines_present[i] + num_lines_present[j])

                            num_lines_present[i] = num_lines_present[i] + num_lines_present[j]
                            num_lines_present[j] = 0
                        else:
                            # Calculating the average end points, middle points and angles of the weighted lines and
                            # put it directly in the lines array. With weighted I mean how many lines are already
                            # put at the two spots which we will merge.
                            lines[j][0][0] = (lines[i][0][0] * num_lines_present[i] + lines[j][0][0] *
                                              num_lines_present[j]) / (num_lines_present[i] + num_lines_present[j])
                            lines[j][0][1] = (lines[i][0][1] * num_lines_present[i] + lines[j][0][1] *
                                              num_lines_present[j]) / (num_lines_present[i] + num_lines_present[j])
                            lines[j][0][2] = (lines[i][0][2] * num_lines_present[i] + lines[j][0][2] *
                                              num_lines_present[j]) / (num_lines_present[i] + num_lines_present[j])
                            lines[j][0][3] = (lines[i][0][3] * num_lines_present[i] + lines[j][0][3] *
                                              num_lines_present[j]) / (num_lines_present[i] + num_lines_present[j])
                            angle_lines[j] = (angle_lines[i] * num_lines_present[i]
                                              + angle_lines[j] * num_lines_present[j]) / \
                                             (num_lines_present[i] + num_lines_present[j])
                            x_lines[j] = (x_lines[i] * num_lines_present[i] + x_lines[j] * num_lines_present[
                                j]) / (num_lines_present[i] + num_lines_present[j])
                            y_lines[j] = (y_lines[i] * num_lines_present[i] + y_lines[j] * num_lines_present[
                                j]) / (num_lines_present[i] + num_lines_present[j])
                            num_lines_present[j] = num_lines_present[i] + num_lines_present[j]
                            num_lines_present[i] = 0

    # here select for the lines that are left after combining
    lines = lines[num_lines_present > 0]
    x_lines = x_lines[num_lines_present > 0]
    y_lines = y_lines[num_lines_present > 0]
    angle_lines = angle_lines[num_lines_present > 0]

    return x_lines, y_lines, lines, angle_lines


def group_single_lines(x_lines, y_lines, lines, angle_lines, max_distance, max_angle_diff=np.pi / 12):
    """
    Here individual lines are grouped together based on their angle and perpendicular distance between each other.
    The lines here are regarded as lines with infinite length so that always the perpendicular distance between lines
    can be calculated.

    Parameters:
        x_lines (ndarray):              The x-centers of the lines.
        y_lines (ndarray):              The y-centers of the lines.
        lines (ndarray):                The end positions of the lines.
        angle_lines (ndarray):          The angles of the lines.
        max_distance (float):           The maximum perpendicular distance between lines in pixels to group them.
        max_angle_diff (float):         The maximum difference in angles between lines to group them.

    Returns:
        groups (list):                  A list with all the groups created. Each group has the index values of the
                                        lines in that group. So lines[groups[0]] gives you the lines of the first group.
    """

    if lines is None:
        return None

    groups = []
    # for each line we look if it fits in a group, if not, we make a new group with that line as a start
    for i in range(len(lines)):
        set_in_group = False
        x1 = x_lines[i]
        y1 = y_lines[i]
        angle = angle_lines[i]
        for j in range(len(groups)):
            angle_mean = np.mean(angle_lines[groups[j]])
            x2 = np.mean(x_lines[groups[j]])
            y2 = np.mean(y_lines[groups[j]])
            if np.abs(angle - angle_mean) <= max_angle_diff:
                x3 = (y2 - y1 + np.tan(angle_mean + np.pi / 2 + 1e-10) * x1 - np.tan(angle_mean + 1e-10) * x2) \
                     / (np.tan(angle_mean + np.pi / 2 + 1e-10) - np.tan(angle_mean + 1e-10))
                y3 = np.tan(angle_mean + 1e-10) * (x3 - x2) + y2
                dist = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
                if dist <= max_distance:  # img_after.shape[1] / 20
                    set_in_group = True
                    groups[j].append(i)
        if not set_in_group:
            groups.append([i])

    for i in range(len(groups)):
        groups[i] = np.unique(groups[i])

    return groups


def couple_groups_of_lines_bas(groups, x_lines, y_lines, angle_lines, min_dist, max_dist,
                               max_angle_diff=np.pi / 8):
    """
    FROM "OLD" VERSION:
    Here the function will look if groups of lines can together form a band which has roughly the same width as the
    milling site. If multiple groups of lines can form bands, the one with the most individual lines will be outputted.
    If these bands have the same number of lines in them, the function looks if they can be merged or not, if not,
    the band closest to the estimated milling site will be used.

    Parameters:
        groups (list):              The groups of individual lines outputted from group_single_lines.
        x_lines (ndarray):          The x-centers of the lines.
        y_lines (ndarray):          The y-centers of the lines.
        angle_lines (ndarray):      The angles of the lines.
        min_dist (float):           The minimum perpendicular distance between groups of lines in pixels to group them.
        max_dist (float):           The maximum perpendicular distance between groups of lines in pixels to group them.
        max_angle_diff (float):     The maximum difference in angles between groups of lines to group them.

    Returns:
        after_grouping (ndarray):   The index values of the lines of the biggest group after the grouping of groups.
                                    These lines (probably) represent the edges of the milling site.

    """

    if groups is None:
        return None

    if len(groups) == 0:
        return np.array([])

    size_groups_combined = np.zeros((len(groups), len(groups)))
    # here we look at each combination of groups and if they have the right conditions, if so, the combined size of
    # the two groups will be put in the output array.
    for i in range(len(groups)):
        for j in np.arange(i + 1, len(groups), 1):
            x1 = np.mean(x_lines[groups[i]])
            y1 = np.mean(y_lines[groups[i]])
            x2 = np.mean(x_lines[groups[j]])
            y2 = np.mean(y_lines[groups[j]])
            dist_centers = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if 1.5 * max_dist >= dist_centers > min_dist / 1.5 and \
                    np.abs(np.mean(angle_lines[groups[i]]) - np.mean(angle_lines[groups[j]])) <= max_angle_diff:
                angle_mean = (np.mean(angle_lines[groups[i]]) * len(groups[i])
                              + np.mean(angle_lines[groups[j]]) * len(groups[j])) / \
                             (len(groups[i]) + len(groups[j]))

                x3 = (y2 - y1 + np.tan(angle_mean + np.pi / 2 + 1e-10) * x1 - np.tan(angle_mean + 1e-10) * x2) / \
                     (np.tan(angle_mean + np.pi / 2 + 1e-10) - np.tan(angle_mean + 1e-10))
                y3 = np.tan(angle_mean + 1e-10) * (x3 - x2) + y2
                dist = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)

                if (dist < max_dist) & (dist > min_dist):
                    size_groups_combined[i, j] = (len(groups[i]) + len(groups[j]))

    # if there are some groups combined
    if np.max(size_groups_combined) > 0:
        index = np.where(size_groups_combined > 0)
        after_grouping = []
        for q in range(len(index[0])):
            for w in groups[index[0][q]]:
                after_grouping.append(w)
            for w in groups[index[1][q]]:
                after_grouping.append(w)
        after_grouping = np.unique(after_grouping)

    else:  # there are no coupled line groups
        after_grouping = np.array([])

    return after_grouping


def detect_blobs_bas(binary_end_result, min_thres=100, max_thres=150, min_circ=0, max_circ=1, min_area=0,
                     max_area=np.inf, min_in=0, max_in=1,
                     min_con=0, max_con=1, plotting=False):
    """
    Here blob detection from opencv is performed on a binary image (or an image with intensities ranging from 0 to 1).
    You can set the constraints of the blob detection with the parameters of this function.

    If you want more information on the opencv blob detection and how the parameters exactly work:
    https://learnopencv.com/blob-detection-using-opencv-python-c/

    Parameters:
        binary_end_result (ndarray):   The image on which blob detection should be performed.
        min_circ (float):               The minimal circularity the blob needs to have to be detected (range:[0,1])
        max_circ (float):               The maximal circularity the blob needs to have to be detected (range:[0,1])
        min_area (float):               The minimal area in pixels the blob needs to have to be detected (range:[0,inf])
        max_area (float):               The maximal area in pixels the blob needs to have to be detected (range:[0,inf])
        min_in (float):                 The minimal inertia the blob needs to have to be detected (range:[0,1])
        max_in (float):                 The maximal inertia the blob needs to have to be detected (range:[0,1])
        min_con (float):                The minimal convexity the blob needs to have to be detected (range:[0,1])
        max_con (float):                The maximal convexity the blob needs to have to be detected (range:[0,1])
        plotting (bool):                If set to True, the detected blobs are shown with red circles in the image.

    Returns:
        key_points (tuple):             The raw output of the opencv blob detection.
        yxr (ndarray):                  Only the y,x position and the radius of the blob in a numpy array.

    """

    # converting the image to a format the opencv blob detection can use
    im = np.array(binary_end_result * 255,
                  dtype=np.uint8)  # cv2.cvtColor(binary_end_result2.astype('uint8'), cv2.COLOR_BGR2GRAY)
    im = cv2.cvtColor(im, cv2.IMREAD_GRAYSCALE)

    # setting the parameters of the blob detections
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = False
    # params.minThreshold = min_thres
    # params.maxThreshold = max_thres

    # circularity
    if (min_circ == 0) & (max_circ == 1):
        params.filterByCircularity = False
    else:
        params.filterByCircularity = True
        params.minCircularity = min_circ
        params.maxCircularity = max_circ

    # area
    if (min_area == 0) & (max_area == np.inf):
        params.filterByArea = False
    else:
        params.filterByArea = True
        params.minArea = min_area
        params.maxArea = max_area

    # inertia
    if (min_in == 0) & (max_in == 1):
        params.filterByInertia = False
    else:
        params.filterByInertia = True
        params.minInertiaRatio = min_in
        params.maxInertiaRatio = max_in

    # convexity
    if (min_con == 0) & (max_con == 1):
        params.filterByConvexity = False
    else:
        params.filterByConvexity = True
        params.minConvexity = min_con
        params.maxConvexity = max_con

    # creating the detector with the specified parameters
    ver = cv2.__version__.split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    # detecting the blobs
    key_points = detector.detect(im)

    # reading out the key_points and putting them in a numpy array
    yxr = np.zeros((len(key_points), 3), dtype=int)
    for u in range(len(key_points)):
        yxr[u, 1] = int(key_points[u].pt[0])
        yxr[u, 0] = int(key_points[u].pt[1])
        yxr[u, 2] = int(key_points[u].size / 2)  # the size is the diameter

    # Draw them if specified
    if plotting:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
        for center_y, center_x, radius in zip(yxr[:, 0], yxr[:, 1], yxr[:, 2]):
            circy, circx = circle_perimeter(center_y, center_x, radius,
                                            shape=im.shape)
            im[circy, circx] = (220, 20, 20, 255)
            im[circy + 1, circx] = (220, 20, 20, 255)
            im[circy, circx + 1] = (220, 20, 20, 255)
            im[circy + 1, circx + 1] = (220, 20, 20, 255)
            im[circy - 1, circx] = (220, 20, 20, 255)
            im[circy, circx - 1] = (220, 20, 20, 255)
            im[circy - 1, circx - 1] = (220, 20, 20, 255)

        ax.imshow(im)
        plt.show()

    return key_points, yxr


data = tiff.read_data("/home/victoria/Downloads/fastem_overview_9.tif")
img = np.array(data[0])
img = gaussian_filter(img, 1)
downscale_factor = 4
img_rescaled1 = np.zeros((img.shape[0], int(img.shape[1] / downscale_factor + 1)))
img_rescaled2 = np.zeros((int(img.shape[0] / downscale_factor + 1), int(img.shape[1] / downscale_factor + 1)))

# interpolate for the y values of the image
x_vals = np.arange(0, img.shape[1], 1)
x_new = np.arange(0, img.shape[1], downscale_factor)

for i in range(img_rescaled1.shape[0]):
    y_vals = img[i, :]
    y_new = np.interp(x_new, x_vals, y_vals)
    img_rescaled1[i, :] = y_new

# interpolate for the x values of the image
y_vals = np.arange(0, img.shape[0], 1)
y_new = np.arange(0, img.shape[0], downscale_factor)
for i in range(img_rescaled1.shape[1]):
    x_vals = img_rescaled1[:, i]
    x_new = np.interp(y_new, y_vals, x_vals)
    img_rescaled2[:, i] = x_new

print(img_rescaled2.shape)
img = np.array(img_rescaled2)

del data, img_rescaled1, img_rescaled2, x_vals, x_new, y_vals, y_new
gc.collect()

histo, bins = np.histogram(img, bins=1000)
plt.plot(bins[:-1], histo)
plt.show()

img_blurred = gaussian_filter(img, 2)
b = img_blurred > 55000  # 55000 for 9; 2e4 for 5
b = binary_closing(b, iterations=3)
b = binary_opening(b, iterations=25)
b = binary_dilation(b, iterations=100)

mask = 1 - 1 * b
mask = np.array(mask, dtype=bool)
img = img * mask

low_thres = 37000  # 37000, 8000
high_thres = 50000  # 50000, 23000
img = (img - low_thres) / (high_thres - low_thres)
img[img > 1] = 1
img[img < 0] = 0

img_blurred = (img_blurred - low_thres) / (high_thres - low_thres)
img_blurred[img_blurred > 1] = 1
img_blurred[img_blurred < 0] = 0

x_lines, y_lines, lines, edges = find_lines_bas(img, blur=0, low_thres_edges=0.03, high_thres_edges=0.05,
                                                angle_res=np.pi / 180, line_thres=1, min_len_lines=1 / 64,
                                                max_line_gap=1 / 102)

angle_lines = calculate_angles(lines)
x_lines2, y_lines2, lines2, angle_lines2 = combine_and_constraint_lines(x_lines, y_lines, lines, angle_lines,
                                                                        img.shape[1] / 2, img.shape,
                                                                        x_width_constraint=1, y_width_constraint=1,
                                                                        angle_constraint=np.pi / 200,
                                                                        max_dist=10 / 2559,
                                                                        max_diff_angle=np.pi / 10)

groups = group_single_lines(x_lines2, y_lines2, lines2, angle_lines2, max_distance=img.shape[1] / 10000)
after_grouping = couple_groups_of_lines_bas(groups, x_lines2, y_lines2, angle_lines2,
                                            max_dist=70, min_dist=50, max_angle_diff=np.pi / 8)
# edges = cv2.Canny(np.array(img, dtype=np.uint8), 0.05, 0.1)

# edges = binary_closing(edges, iterations=1)
# edges = binary_opening(edges, iterations=1)

# lines = cv2.HoughLinesP(np.array(edges, dtype=np.uint8), 1, np.pi/180, threshold=1, minLineLength=100, maxLineGap=50)

del edges, x_lines, y_lines
gc.collect()

all_lines_img = np.zeros(img.shape)
comb_lines_img = np.zeros(img.shape)
after_lines_img = np.zeros(img.shape)

if lines is not None:
    for points in lines:
        # Extracted points nested in the list
        x1, y1, x2, y2 = points[0]
        # Draw the lines joining the points on the original image
        cv2.line(all_lines_img, (x1, y1), (x2, y2), 1, 2)

    for points in lines2:
        # Extracted points nested in the list
        x1, y1, x2, y2 = points[0]
        # Draw the lines joining the points on the original image
        cv2.line(comb_lines_img, (x1, y1), (x2, y2), 1, 2)

    for points in lines2[after_grouping]:
        # Extracted points nested in the list
        x1, y1, x2, y2 = points[0]
        # Draw the lines joining the points on the original image
        cv2.line(after_lines_img, (x1, y1), (x2, y2), 1, 2)

print("now plotting")
# plt.imshow(lines_img)
# img = img / np.max(img)
# lines_img = lines_img / np.max(lines_img)
fig, ax = plt.subplots(ncols=3, sharex=True, sharey=True)
ax[0].imshow(img + all_lines_img)
ax[1].imshow(img + comb_lines_img)
ax[2].imshow(img + after_lines_img)
plt.show()

print("you can try now something else")

keypoints, yxr = detect_blobs_bas(img_blurred * mask, min_thres=100, max_thres=250, min_circ=0.65, max_circ=0.9,
                                  min_area=2000, max_area=10000, min_in=0.6, max_in=1,
                                  min_con=0.6, max_con=1, plotting=True)

print("")

keypoints2, yxr2 = detect_blobs_bas(img_blurred * mask, min_thres=100, max_thres=250, min_circ=0, max_circ=0.5,
                                  min_area=30 * 50 * 50, max_area=200 * 50 * 50,
                                  min_in=0, max_in=0.5, min_con=0, max_con=1, plotting=True)

print("you can try now something else")

binary_img = mask * gaussian_filter(img_blurred, sigma=5) * 255 > 100
b1 = binary_closing(binary_img, iterations=3)
b2 = binary_opening(b1, iterations=20)

plt.imshow(b2)
plt.show()
