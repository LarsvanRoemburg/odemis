import numpy as np
import matplotlib.pyplot as plt
from odemis.dataio import tiff
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import binary_erosion, binary_opening, binary_closing
from scipy.signal import fftconvolve

# import analyse_shifts

which_path = 1
threshold_mask = 0.25
threshold_end = 0.3
blur = 25
cropping = True

channel_before = [0, 0, 0, 0, 0, 0, 0, 1]
channel_after = [0, 0, 0, 0, 0, 0, 0, 0]
thr_out_before = [300, 700, 3000, 300, 300, 300, 300, 200]
thr_out_after = [300, 700, 3000, 300, 300, 300, 300, 8000]
z_value_before = [20, 23, 20, 25, 30, np.pi, 25, np.pi]  # np.pi means it is not a z-stack, only one image
z_value_after = [13, 15, 20, 19, 30, np.pi, 19, 1]  # np.pi means it is not a z-stack, only one image

data_paths_before = ["/home/victoria/Documents/Lars/data/1/FOV2_GFP_cp00.tif",
                     "/home/victoria/Documents/Lars/data/2/FOV1_checkpoint_00.tiff",
                     "/home/victoria/Documents/Lars/data/3/FOV4_checkpoint_00.tiff",
                     "/home/victoria/Documents/Lars/data/4/METEOR_images/FOV3_checkpoint_00.tif",
                     "/home/victoria/Documents/Lars/data/6/20201002_FOV2_checkpoint_001_stack_001.tiff",
                     "/home/victoria/Documents/Lars/data/7/FOV3_Meteor_stacks/Lamella_slices_tif"
                     "/EA010_8_FOV3_checkpoint_00_1-24.tif",
                     "/home/victoria/Documents/Lars/data/meteor1/FOV3_Meteor_stacks/EA010_8_FOV3_checkpoint_00_1.tiff",
                     "/home/victoria/Documents/Lars/data/XA Yeast-20220215T093027Z-001/XA "
                     "Yeast/20200918_millingsite_start.ome.tiff"
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
                    "Yeast/20200918-zstack_400nm.ome.tiff"
                    ]

data_before_milling = tiff.read_data(data_paths_before[which_path])
data_after_milling = tiff.read_data(data_paths_after[which_path])

meta_before = data_before_milling[channel_before[which_path]].metadata
meta_after = data_after_milling[channel_after[which_path]].metadata

print("Pixel sizes are the same: {}".format(meta_before["Pixel size"][0] == meta_after["Pixel size"][0]))

# convert the image to a numpy array
if z_value_before[which_path] != np.pi:
    img_before = np.array(data_before_milling[channel_before[which_path]][0][0][z_value_before[which_path]],
                      dtype=int)
else:
    img_before = np.array(data_before_milling[channel_before[which_path]], dtype=int)

if z_value_after[which_path] != np.pi:
    img_after = np.array(data_after_milling[channel_after[which_path]][0][0][z_value_after[which_path]],
                         dtype=int)
else:
    img_after = np.array(data_after_milling[channel_after[which_path]], dtype=int)

# setting a threshold for outliers in the image
img_before[img_before > thr_out_before[which_path]] = thr_out_before[which_path]
img_after[img_after > thr_out_after[which_path]] = thr_out_after[which_path]

# rescaling one image to the other if necessary
if img_after.shape != img_before.shape:
    if img_after.shape > img_before.shape:
        img_rescaled = np.zeros(img_after.shape)
        img_small = img_before
        d_pix = img_before.shape[0] / img_after.shape[0]  # here we assume that the difference in x is the same as in
        # y as pixels are squares

    else:
        img_rescaled = np.zeros(img_before.shape)
        img_small = img_after
        img_big = img_before.shape
        d_pix = img_after.shape[0] / img_before.shape[0]  # here we assume that the difference in x is the same as in
        # y as pixels are squares

    for i in range(img_small.shape[0]):
        x_vals = np.arange(0, len(img_small[i, :]), 1)
        y_vals = img_small[i, :]
        x_new = np.arange(0, len(img_small[i, :]), d_pix)
        y_new = np.interp(x_new, x_vals, y_vals)
        img_rescaled[i, :] = y_new

    for i in range(img_rescaled.shape[1]):
        y_vals = np.arange(0, len(img_small[:, 0]), 1)
        x_vals = img_rescaled[:img_small.shape[0], i]
        y_new = np.arange(0, len(img_small[:, 0]), d_pix)
        x_new = np.interp(y_new, y_vals, x_vals)
        img_rescaled[:, i] = x_new

    if img_after.shape > img_before.shape:
        img_before = img_rescaled
    else:
        img_after = img_rescaled

# calculate the shift between the two images
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

# preprocessing steps of the images: blurring the image
img_after_blurred = gaussian_filter(img_after, sigma=blur)
img_before_blurred = gaussian_filter(img_before, sigma=blur)

print(np.min(img_before_blurred))
print(np.min(img_after_blurred))
print(np.max(img_before_blurred))
print(np.max(img_after_blurred))
print("\n")

# preprocessing steps of the images: normalization to [0,1] interval
base_lvl_before = np.min(img_before_blurred)
base_lvl_after = np.min(img_after_blurred)
img_before_blurred = (img_before_blurred - base_lvl_before) / (np.max(img_before_blurred) - base_lvl_before)
img_after_blurred = (img_after_blurred - base_lvl_after) / (np.max(img_after_blurred) - base_lvl_after)

img_before = (img_before - base_lvl_before) / (np.max(img_before) - base_lvl_before)
img_after = (img_after - base_lvl_after) / (np.max(img_after) - base_lvl_after)
img_before[img_before < 0] = 0
img_after[img_after < 0] = 0

# calculating the difference between the two images and creating a mask
diff = img_before_blurred - 2 * img_after_blurred  # times 2 to account for the intensity differences (more robust)
mask = diff >= threshold_mask
mask2 = binary_opening(mask, iterations=int(blur/2))  # First opening
# mask2 = binary_erosion(mask2, iterations=5)  # if the edges give too much signal we can erode the mask a bit more

masked_img = img_after * mask
masked_img2 = img_after * mask2

# to crop the image around the ROI if possible and wanted: cropping = True
index_mask = np.where(mask)

if cropping & (len(index_mask[0]) != 0):
    x_min = np.min(index_mask[1])
    y_min = np.min(index_mask[0])
    x_max = np.max(index_mask[1])
    y_max = np.max(index_mask[0])

    masked_img = masked_img[y_min:y_max, x_min:x_max]
    masked_img2 = masked_img2[y_min:y_max, x_min:x_max]

# setting a threshold for the image_after within the mask
binary_end_result1 = masked_img >= threshold_end
binary_end_result2 = masked_img2 >= threshold_end

# getting rid of too small ROIs and noise after the threshold
# binary_end_result1 = binary_opening(binary_end_result1, iterations=5)
binary_end_result2 = binary_opening(binary_closing(binary_end_result2, iterations=1), iterations=5)  # second opening

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

# plotting a histogram of the intensity values of the differences
# plt.title('histogram')
# plt.xlabel("value")
# plt.ylabel("frequency")
# plt.hist(diff.flatten())
# plt.show()

masked_img[0, 0] = 1  # to give them the same color scale
masked_img2[0, 0] = 1  # to give them the same color scale

# plotting the results
fig, ax = plt.subplots(4, 2)
ax[0, 0].imshow(img_before)
ax[0, 0].set_title("before")
ax[0, 1].imshow(img_after)
ax[0, 1].set_title("after")
ax[1, 0].imshow(img_before_blurred)
ax[1, 1].imshow(img_after_blurred)
if cropping & (len(index_mask[0]) != 0):
    ax[2, 0].imshow(masked_img, extent=(x_min, x_max, y_max, y_min))
    ax[2, 1].imshow(masked_img2, extent=(x_min, x_max, y_max, y_min))
    ax[3, 0].imshow(binary_end_result1, extent=(x_min, x_max, y_max, y_min))
    ax[3, 1].imshow(binary_end_result2, extent=(x_min, x_max, y_max, y_min))
else:
    ax[2, 0].imshow(masked_img)
    ax[2, 1].imshow(masked_img2)
    ax[3, 0].imshow(binary_end_result1)
    ax[3, 1].imshow(binary_end_result2)
ax[2, 0].set_title("masked without opening")
ax[2, 1].set_title("masked with opening")

ax[3, 0].set_title("end without 2nd opening")

ax[3, 1].set_title("end with 2nd opening")
plt.show()

print("Successful!")
