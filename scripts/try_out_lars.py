import numpy as np
import matplotlib.pyplot as plt
from odemis.dataio import tiff
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import binary_erosion

data_before_milling = tiff.read_data("/home/victoria/Documents/Lars/data/3/FOV4_checkpoint_00.tiff")  #  /2/FOV1_checkpoint_00.tiff 1/FOV2_GFP_cp00.tif
data_after_milling = tiff.read_data("/home/victoria/Documents/Lars/data/3/FOV4_checkpoint_01.tiff")  # /2/FOV1_checkpoint_01.tiff 1/FOV2_GFP_cp04.tif

meta_before = data_before_milling[0].metadata
meta_after = data_after_milling[0].metadata

print("Pixel sizes are the same: {}".format(meta_before["Pixel size"] == meta_after["Pixel size"]))

dx = meta_before["Centre position"][0] - meta_after["Centre position"][0]
dy = meta_before["Centre position"][1] - meta_after["Centre position"][1]
dz = meta_before["Centre position"][2] - meta_after["Centre position"][2]
xy_size = meta_before["Pixel size"][0]
z_size = meta_before["Pixel size"][2]
num_slices = 0 #int(dz/z_size)
dx_pix = int(dx/xy_size)
dy_pix = int(dy/xy_size)

print("dx is {} pixels".format(dx_pix))
print("dy is {} pixels".format(dy_pix))
print("dz is {} slices".format(num_slices))

p1 = 20
p2 = p1 + num_slices  # p1 + num_slices  # - 10  # - 7  # there is a 7 slices difference in the z position
blur = 25
threshold_outliers = 3000
threshold_mask = 0.25

img_before = np.array(data_before_milling[0][0][0][p1], dtype=int)  # data_before_milling[0][0][0][17]
img_after = np.array(data_after_milling[0][0][0][p2], dtype=int)  # data_after_milling[0][0][0][17]
img_before[img_before > threshold_outliers] = threshold_outliers
img_after[img_after > threshold_outliers] = threshold_outliers
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

if dx_pix > 0:
    img_after[:, dx_pix:] = img_after[:, :-dx_pix]
elif dx_pix < 0:
    img_after[:, :dx_pix] = img_after[:, -dx_pix:]

if dy_pix > 0:
    img_after[dy_pix:, :] = img_after[:-dy_pix, :]
elif dy_pix < 0:
    img_after[:dy_pix, :] = img_after[-dy_pix:, :]


'''
test = np.zeros((500, 500))
test[250, 250] = 100
test = gaussian_filter(test, sigma=25)
test = test/np.max(test) > 0.01
fig, ax = plt.subplots(5, 1)
ax[0].imshow(test)
test = binary_erosion(test, iterations=10)  # , structure=np.ones((3, 3)))
ax[1].imshow(test)
test = binary_erosion(test, iterations=10)  # , structure=np.ones((3, 3)))
ax[2].imshow(test)
test = binary_erosion(test, iterations=10)  # , structure=np.ones((3, 3)))
ax[3].imshow(test)
test = binary_erosion(test, iterations=10)  # , structure=np.ones((3, 3)))
ax[4].imshow(test)
plt.show()
'''
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
base_lvl_cp0 = np.min(img_before_blurred)
base_lvl_cp4 = np.min(img_after_blurred)
img_before_blurred = (img_before_blurred - base_lvl_cp0) / (np.max(img_before_blurred) - base_lvl_cp0)
img_after_blurred = (img_after_blurred - base_lvl_cp4) / (np.max(img_after_blurred) - base_lvl_cp4)

diff = img_before_blurred - img_after_blurred
mask = diff >= threshold_mask
mask2 = binary_erosion(mask, iterations=blur)
# mask = erode mask to get rid of signal outside the roi due to blurring
masked_img = img_after_blurred * mask
masked_img2 = img_after_blurred * mask2

masked_img[0, 0] = 1  # to give them the same color scale
masked_img2[0, 0] = 1  # to give them the same color scale

index_mask = np.where(mask == True)
x_min = np.min(index_mask[1])
y_min = np.min(index_mask[0])
x_max = np.max(index_mask[1])
y_max = np.max(index_mask[0])

masked_img_cropped = masked_img[y_min-5:y_max+5, x_min-5:x_max+5]
masked_img2_cropped = masked_img2[y_min-5:y_max+5, x_min-5:x_max+5]

binary_end_result1 = masked_img >= 0.4
binary_end_result2 = masked_img2 >= 0.4

# masked_img[mask] = masked_img[mask] - base_lvl_cp4
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
ax[2, 0].set_title("the difference")
ax[2, 1].imshow(masked_img2)
ax[3, 0].imshow(binary_end_result1)
ax[3, 1].imshow(binary_end_result2)
plt.show()

print("Succesful!")

'''
step = np.linspace(15, 25, 11, dtype = int)
for i in step:
    img = data_before_milling[0][0][0][i]
    img[img>300] = 300
    plt.imshow(img)
    plt.title("z-stack i={}".format(i+15))
    plt.show()

#plt.imshow(img_after)
#plt.show()
'''