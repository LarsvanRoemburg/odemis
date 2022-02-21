import numpy as np
import matplotlib.pyplot as plt
from odemis.dataio import tiff
from scipy.ndimage.filters import gaussian_filter

data_before_milling = tiff.read_data("/home/victoria/Documents/Lars/data/1/FOV2_GFP_cp00.tif")
data_after_milling = tiff.read_data("/home/victoria/Documents/Lars/data/1/FOV2_GFP_cp04.tif")

p1 = 17
p2 = p1 - 7  # there is a 7 slices difference in the z position
blur = 25
threshold_outliers = 300
threshold_mask = 0.25

img_cp0 = np.array(data_before_milling[0][0][0][p1], dtype=int)  # data_before_milling[0][0][0][17]
img_cp4 = np.array(data_after_milling[0][0][0][p2], dtype=int)  # data_after_milling[0][0][0][17]

print(np.min(img_cp0))
print(np.min(img_cp4))
print("\n")
print(np.max(img_cp0))
print(np.max(img_cp4))
print("\n")

img_cp0[img_cp0 > threshold_outliers] = threshold_outliers
img_cp4[img_cp4 > threshold_outliers] = threshold_outliers
img_cp4_blurred = gaussian_filter(img_cp4, sigma=blur)
img_cp0_blurred = gaussian_filter(img_cp0, sigma=blur)

print(np.min(img_cp0_blurred))
print(np.min(img_cp4_blurred))
print(np.max(img_cp0_blurred))
print(np.max(img_cp4_blurred))
print("\n")

# normalization to [0,1] interval
base_lvl_cp0 = np.min(img_cp0_blurred)
base_lvl_cp4 = np.min(img_cp4_blurred)
img_cp0_blurred = (img_cp0_blurred - base_lvl_cp0) / (np.max(img_cp0_blurred) - base_lvl_cp0)
img_cp4_blurred = (img_cp4_blurred - base_lvl_cp4) / (np.max(img_cp4_blurred) - base_lvl_cp4)

diff = img_cp0_blurred - img_cp4_blurred
mask = diff >= threshold_mask
masked_img = img_cp4_blurred * mask
# masked_img[mask] = masked_img[mask] - base_lvl_cp4
# masked_img[masked_img < 0] = 0

print(np.min(img_cp0))
print(np.min(img_cp4))
print(np.min(img_cp0_blurred))
print(np.min(img_cp4_blurred))
print(np.min(diff))
print("\n")
print(np.max(img_cp0))
print(np.max(img_cp4))
print(np.max(img_cp0_blurred))
print(np.max(img_cp4_blurred))
print(np.max(diff))


# plt.title('histogram')
# plt.xlabel("value")
# plt.ylabel("frequency")
# plt.hist(diff.flatten())
# plt.show()


fig, ax = plt.subplots(3, 2)
ax[0, 0].imshow(img_cp0)
ax[0, 0].set_title("before")
ax[0, 1].imshow(img_cp4)
ax[0, 1].set_title("after")
ax[1, 0].imshow(img_cp0_blurred)
ax[1, 1].imshow(img_cp4_blurred)
ax[2, 0].imshow(diff)
ax[2, 0].set_title("the difference")
ax[2, 1].imshow(masked_img)
plt.show()

'''
step = np.linspace(15, 25, 11, dtype = int)
for i in step:
    img = data_before_milling[0][0][0][i]
    img[img>300] = 300
    plt.imshow(img)
    plt.title("z-stack i={}".format(i+15))
    plt.show()

#plt.imshow(img_cp4)
#plt.show()
'''