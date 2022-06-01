import gc
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.feature import match_template, peak_local_max
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale, rotate

from functions_lars import z_projection_and_outlier_cutoff, blur_and_norm
from odemis.dataio import tiff


def crop_to_border(img, coordinates=None):
    x1, y1, x2, y2 = coordinates
    cropped = img[y1:y2, x1:x2]

    return cropped


def bounding_box(cnt, offset=0):    # TODO: add state cnt length check
    x1 = np.min(np.array(cnt)[::, 0]) - offset
    y1 = np.min(np.array(cnt)[::, 1]) - offset
    x2 = np.max(np.array(cnt)[::, 0]) + offset
    y2 = np.max(np.array(cnt)[::, 1]) + offset

    return x1, y1, x2, y2


def show_match(image, template, result, start_point, threshold_rel=None, threshold_abs=None, min_distance=1):
    i = 1

    plt.imshow(image)
    # rect = plt.Rectangle(start_point, template.shape[1], template.shape[0], color='r', fc='none')
    # plt.gca().add_patch(rect)
    for y, x in peak_local_max(result, threshold_rel=threshold_rel, threshold_abs=threshold_abs,
                               min_distance=min_distance):
        rect = plt.Rectangle((x, y), template.shape[1], template.shape[0], color=0.5 * np.random.rand(3, ) + 0.5,
                             fc='none')
        circle = plt.Circle((x + template.shape[1] // 2, y + template.shape[0] // 2), 1, color='red')

        plt.gca().text(x + 10 + template.shape[1] // 2, y + 10 + template.shape[0] // 2, i,
                       color=(max((1 - i / 100), 0.1), 0, 0), fontsize=8)
        plt.gca().add_patch(circle)
        i += 1
    plt.show()


def draw_section_contour(image):
    select = PointSelector(image)
    select.run()
    return select.points


class PointSelector(object):
    def __init__(self, img):
        self.image = img
        self.name = ' '
        self.finished = False
        self.points = []
        self.pos = (0, 0)

    def on_event(self, event, x, y, flags, param):
        if self.finished:
            return
        elif event == cv2.EVENT_LBUTTONUP:
            self.points.append([x, y])
        elif event == cv2.EVENT_MOUSEMOVE:
            self.pos = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.finished = True

    def run(self):
        cv2.namedWindow(self.name)
        cv2.setMouseCallback(self.name, self.on_event)
        copy = self.image.copy()

        while not self.finished:
            copy = self.image.copy()
            if len(self.points) > 0:
                cv2.line(copy, self.points[-1], self.pos, (150, 150, 150), thickness=1)
                cv2.polylines(copy, [np.array([self.points])], self.finished, (150, 150, 150), thickness=1)
            else:
                pass
            cv2.imshow(self.name, copy)
            cv2.waitKey(1)

        cv2.polylines(copy, [np.array([self.points])], self.finished, (150, 150, 150), thickness=1)
        cv2.imshow(self.name, copy)
        cv2.waitKey()
        cv2.destroyAllWindows()


directory = "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Mammalian_cells/"
single_test_img = 'FOV11_after_GFP.tif'

path = os.path.join(directory, single_test_img)
data = tiff.read_data(path)
img = z_projection_and_outlier_cutoff(data, 30, 0, mode='max')
#rescale(data[0], 0.2, True)
del data
gc.collect()
img, img_blurred = blur_and_norm(img, blur=25)

threshold = 0.01

img = gaussian_filter(img, 1)

# x1, y1, x2, y2 = bounding_box(draw_section_contour(img), offset=10)
# template = crop_to_border(img, [x1, y1, x2, y2])
template = np.zeros((500, 500))
template[:, :151] = 1
template[:, -150:] = 1
# plt.imshow(template)
# plt.show()
result = match_template(img, template)
total = np.zeros(result.shape)

start_angle = -4
stop_angle = 4
angle_step = 0.5
angles = np.arange(start_angle, stop_angle, angle_step)
total = np.zeros((len(angles), result.shape[0], result.shape[1]))
for a in range(len(angles)):
    result = match_template(img, rotate(template, angles[a], mode='constant', cval=1))
    total[a, :, :] = result
    print("Adding angle results")
    print(angles[a])

#total = total / len(np.arange(start_angle, stop_angle, angle_step))
# plt.imshow(total)
# plt.show()
print(np.where(total == np.max(total)))
show_match(img, template, total, [], threshold_abs=threshold, min_distance=np.min(template.shape) // 3)
