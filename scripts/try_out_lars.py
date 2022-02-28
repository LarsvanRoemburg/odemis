import numpy as np
import matplotlib.pyplot as plt
from odemis.dataio import tiff
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import binary_opening
from scipy.signal import fftconvolve

# import analyse_shifts

which_path = 0
threshold_mask = 0.25
threshold_end = 0.3
blur = 25
cropping = True

thr_out = [300, 700, 3000, 300, 300]
z_value_before = [20, 23, 20, 25, 30]
z_value_after = [13, 15, 20, 19, 30]

data_paths_before = ["/home/victoria/Documents/Lars/data/1/FOV2_GFP_cp00.tif",
                     "/home/victoria/Documents/Lars/data/2/FOV1_checkpoint_00.tiff",
                     "/home/victoria/Documents/Lars/data/3/FOV4_checkpoint_00.tiff",
                     "/home/victoria/Documents/Lars/data/4/METEOR_images/FOV3_checkpoint_00.tif",
                     "/home/victoria/Documents/Lars/data/6/20201002_FOV2_checkpoint_001_stack_001.tiff"
                     ]

data_paths_after = ["/home/victoria/Documents/Lars/data/1/FOV2_GFP_cp04.tif",
                    "/home/victoria/Documents/Lars/data/2/FOV1_checkpoint_01.tiff",
                    "/home/victoria/Documents/Lars/data/3/FOV4_checkpoint_01.tiff",
                    "/home/victoria/Documents/Lars/data/4/METEOR_images/FOV3_checkpoint_01.tif",
                    "/home/victoria/Documents/Lars/data/6/20201002_FOV2_checkpoint_005_stack_001.tiff"
                    ]

data_before_milling = tiff.read_data(data_paths_before[which_path])
data_after_milling = tiff.read_data(data_paths_after[which_path])

meta_before = data_before_milling[0].metadata
meta_after = data_after_milling[0].metadata

print("Pixel sizes are the same: {}".format(meta_before["Pixel size"] == meta_after["Pixel size"]))

'''
dx = meta_before["Centre position"][0] - meta_after["Centre position"][0]
dy = meta_before["Centre position"][1] - meta_after["Centre position"][1]
dz = meta_before["Centre position"][2] - meta_after["Centre position"][2]
xy_size = meta_before["Pixel size"][0]
z_size = meta_before["Pixel size"][2]
dx_pix = int(dx / xy_size)
dy_pix = int(dy / xy_size)
dz_pix = int(dz / z_size)

print("dx is {} pixels".format(dx_pix))
print("dy is {} pixels".format(dy_pix))
print("dz is {} slices".format(dz_pix))
'''

img_before = np.array(data_before_milling[0][0][0][z_value_before[which_path]],
                      dtype=int)  # data_before_milling[0][0][0][17]
img_after = np.array(data_after_milling[0][0][0][z_value_after[which_path]],
                     dtype=int)  # data_after_milling[0][0][0][17]
img_before[img_before > thr_out[which_path]] = thr_out[which_path]
img_after[img_after > thr_out[which_path]] = thr_out[which_path]

if img_after.shape != img_before.shape:
    img_after_rescaled = np.zeros(img_before.shape)

    for i in range(img_after.shape[0]):
        x_vals = np.arange(0, len(img_after[i, :]), 1)
        y_vals = img_after[i, :]
        x_new = np.arange(0, len(img_after[i, :]), 0.5)
        y_new = np.interp(x_new, x_vals, y_vals)
        img_after_rescaled[i, :] = y_new

    for i in range(img_after_rescaled.shape[1]):
        y_vals = np.arange(0, len(img_after[:, 0]), 1)
        x_vals = img_after_rescaled[:img_after.shape[0], i]
        y_new = np.arange(0, len(img_after[:, 0]), 0.5)
        x_new = np.interp(y_new, y_vals, x_vals)
        img_after_rescaled[:, i] = x_new

    img_after = img_after_rescaled

im1 = img_before
im2 = img_after
im1 = im1 - np.mean(im1)
im2 = im2 - np.mean(im2)

conv = fftconvolve(im1, im2[::-1, ::-1], mode='same')
shift = np.where(conv == np.max(conv))
shift = np.asarray(shift)
shift[0] = shift[0] - img_after.shape[0] / 2
shift[1] = shift[1] - img_after.shape[1] / 2

# shift2 = analyse_shifts.measure_shift(im1, im2, use_md=False)

dy_pix = int(shift[0])
dx_pix = int(shift[1])

print("dx is {} pixels".format(dx_pix))
print("dy is {} pixels".format(dy_pix))
# print("dz is {} slices".format(dz_pix))

# plt.imshow(conv)
# plt.show()

if dx_pix > 0:
    img_after[:, dx_pix:] = img_after[:, :-dx_pix]
elif dx_pix < 0:
    img_after[:, :dx_pix] = img_after[:, -dx_pix:]

if dy_pix > 0:
    img_after[dy_pix:, :] = img_after[:-dy_pix, :]
elif dy_pix < 0:
    img_after[:dy_pix, :] = img_after[-dy_pix:, :]

print(np.min(img_before))
print(np.min(img_after))
print("\n")
print(np.max(img_before))
print(np.max(img_after))
print("\n")

img_after_blurred = gaussian_filter(img_after, sigma=blur)
img_before_blurred = gaussian_filter(img_before, sigma=blur)

print(np.min(img_before_blurred))
print(np.min(img_after_blurred))
print(np.max(img_before_blurred))
print(np.max(img_after_blurred))
print("\n")

# normalization to [0,1] interval
base_lvl_before = np.min(img_before_blurred)
base_lvl_after = np.min(img_after_blurred)
img_before_blurred = (img_before_blurred - base_lvl_before) / (np.max(img_before_blurred) - base_lvl_before)
img_after_blurred = (img_after_blurred - base_lvl_after) / (np.max(img_after_blurred) - base_lvl_after)

img_before = (img_before - base_lvl_before) / (np.max(img_before) - base_lvl_before)
img_after = (img_after - base_lvl_after) / (np.max(img_after) - base_lvl_after)
img_before[img_before < 0] = 0
img_after[img_after < 0] = 0

diff = img_before_blurred - 2*img_after_blurred
mask = diff >= threshold_mask
mask2 = binary_opening(mask, iterations=int(blur/2))  # First opening

# mask = erode mask to get rid of signal outside the roi due to blurring
masked_img = img_after * mask
masked_img2 = img_after * mask2

index_mask = np.where(mask)

if cropping & (len(index_mask[0]) != 0):
    x_min = np.min(index_mask[1])
    y_min = np.min(index_mask[0])
    x_max = np.max(index_mask[1])
    y_max = np.max(index_mask[0])

    masked_img = masked_img[y_min - 5:y_max + 5, x_min - 5:x_max + 5]
    masked_img2 = masked_img2[y_min - 5:y_max + 5, x_min - 5:x_max + 5]

binary_end_result1 = masked_img >= threshold_end
binary_end_result2 = masked_img2 >= threshold_end

# binary_end_result1 = binary_opening(binary_end_result1, iterations=5)
binary_end_result2 = binary_opening(binary_end_result2, iterations=3)  # second opening

# masked_img[mask] = masked_img[mask] - base_lvl_after
# masked_img[masked_img < 0] = 0

print(np.min(img_before))
print(np.min(img_after))
print(np.min(img_before_blurred))
print(np.min(img_after_blurred))
print(np.min(diff))
print("\n")
print(np.max(img_before))
print(np.max(img_after))
print(np.max(img_before_blurred))
print(np.max(img_after_blurred))
print(np.max(diff))

# plt.title('histogram')
# plt.xlabel("value")
# plt.ylabel("frequency")
# plt.hist(diff.flatten())
# plt.show()

masked_img[0, 0] = 1  # to give them the same color scale
masked_img2[0, 0] = 1  # to give them the same color scale

fig, ax = plt.subplots(4, 2)
ax[0, 0].imshow(img_before)
ax[0, 0].set_title("before")
ax[0, 1].imshow(img_after)
ax[0, 1].set_title("after")
ax[1, 0].imshow(img_before_blurred)
ax[1, 1].imshow(img_after_blurred)
ax[2, 0].imshow(masked_img)
ax[2, 0].set_title("masked without opening")
ax[2, 1].imshow(masked_img2)
ax[2, 1].set_title("masked with opening")
ax[3, 0].imshow(binary_end_result1)
ax[3, 0].set_title("end without 2nd opening")
ax[3, 1].imshow(binary_end_result2)
ax[3, 1].set_title("end with 2nd opening")
plt.show()

print("Successful!")
