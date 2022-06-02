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


def downsize_image(image, factor):
    temp_img = np.zeros((int(image.shape[0]/factor), image.shape[1]))
    new_img = np.zeros((int(image.shape[0]/factor), int(image.shape[1]/factor)))

    for e in range(temp_img.shape[0]):
        temp_img[e, :] = np.mean(image[int(factor*e):int(factor*e+factor), :], axis=0)
    for j in range(new_img.shape[1]):
        new_img[:, j] = np.mean(temp_img[:, int(factor*j):int(factor*j+factor)], axis=1)

    return new_img

directory = "/home/victoria/Documents/Lars/data/2/"
single_test_img = 'FOV1_checkpoint_01.tiff'

path = os.path.join(directory, single_test_img)
data = tiff.read_data(path)
img = z_projection_and_outlier_cutoff(data, 30, 0, mode='max')
#rescale(data[0], 0.2, True)
del data
gc.collect()
img, img_blurred = blur_and_norm(img, blur=5)

factor = 4
img = downsize_image(img, factor)
img_blurred = downsize_image(img_blurred, factor)
# threshold = 0.01

# img = gaussian_filter(img, 1)

# x1, y1, x2, y2 = bounding_box(draw_section_contour(img), offset=10)
# template = crop_to_border(img, [x1, y1, x2, y2])
template_size = int(img.shape[0]/2)
num_width = 10
templates = np.zeros((num_width, int(template_size/3), template_size))
widths = np.linspace(img.shape[0]/10, img.shape[0]/4, num_width)
for i in range(num_width):
    border = int(template_size/2 - widths[i]/2)
    templates[i, :, :border] = np.linspace(0, 1, border)
    templates[i, :, -border:] = 1 - np.linspace(0, 1, border)

result = match_template(img, templates[0])

start_angle = -10
stop_angle = 10
angles = np.linspace(start_angle, stop_angle, 11, endpoint=True)
total = np.zeros((len(templates), result.shape[0], result.shape[1]))
for i, template in enumerate(templates):
    result = match_template(img_blurred, template)
    total[i, :, :] = result
    print(f'template nr. {i}')
print(f"best template is nr. {np.where(total == np.max(total))[0][0]}")
best_template = templates[np.where(total == np.max(total))[0][0]]

total = np.zeros((len(angles), result.shape[0], result.shape[1]))
for a, angle in enumerate(angles):
    result = match_template(img_blurred, rotate(best_template, angles[a], mode='constant', cval=1))
    total[a, :, :] = result
    print(f"angle = {angle}")

best_angle = angles[np.where(total == np.max(total))[0][0]]
print(f"best angle = {best_angle}")
best_y = np.where(total == np.max(total))[1][0]
best_x = np.where(total == np.max(total))[2][0]

show_img = np.array(img)
show_img[best_y:best_y+template.shape[0], best_x:best_x+template.shape[1]] += 0.5*rotate(best_template,
                                                                                         best_angle, mode='constant',
                                                                                         cval=0)

total = np.zeros((len(templates), len(angles), result.shape[0], result.shape[1]))
for i, template in enumerate(templates):
    print(f'template nr. {i}')
    for a, angle in enumerate(angles):
        result = match_template(img_blurred, rotate(template, angles[a], mode='constant', cval=0))
        total[i, a, :, :] = result
        print(f"angle = {angle}")

print(np.where(total == np.max(total)))
best_template = templates[np.where(total == np.max(total))[0][0]]
best_angle = angles[np.where(total == np.max(total))[1][0]]
best_y = np.where(total == np.max(total))[2][0]
best_x = np.where(total == np.max(total))[3][0]

#total = total / len(np.arange(start_angle, stop_angle, angle_step))
# plt.imshow(total)
# plt.show()

show_img2 = np.array(img)
show_img2[best_y:best_y+template.shape[0], best_x:best_x+template.shape[1]] += 0.5*rotate(best_template,
                                                                                         best_angle, mode='constant',
                                                                                         cval=0)
fig, ax = plt.subplots(ncols=3)
ax[0].imshow(img)
ax[1].imshow(show_img)
ax[1].set_title("n , m")
ax[2].imshow(show_img2)
ax[2].set_title("n x m")
plt.show()

print("Successful!")
# show_match(img, template, total, [], threshold_abs=threshold, min_distance=np.min(template.shape) // 3)
