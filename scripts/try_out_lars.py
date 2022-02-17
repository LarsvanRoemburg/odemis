import numpy as np
import matplotlib.pyplot as plt
from odemis.dataio import tiff
from scipy.ndimage.filters import gaussian_filter

data_before_milling = tiff.read_data("/home/victoria/Documents/Lars/data/1/FOV2_GFP_cp00.tif")
data_after_milling = tiff.read_data("/home/victoria/Documents/Lars/data/1/FOV2_GFP_cp04.tif")

p1 = 17
p2 = p1 - 7  # there is a 7 slices difference in the z position
blur = 15

img_cp0 = np.array(data_before_milling[0][0][0][p1], dtype=int)  # data_before_milling[0][0][0][17]
img_cp4 = np.array(data_after_milling[0][0][0][p2], dtype=int)  # data_after_milling[0][0][0][17]

print(np.min(img_cp0))
print(np.min(img_cp4))
print("\n")
print(np.max(img_cp0))
print(np.max(img_cp4))
print("\n")

img_cp0[img_cp0 > 300] = 300
img_cp4[img_cp4 > 300] = 300
img_cp4 = gaussian_filter(img_cp4, sigma=blur)
img_cp0 = gaussian_filter(img_cp0, sigma=blur)
diff = img_cp0 - img_cp4
diff[diff<55] = 0
print(np.min(img_cp0))
print(np.min(img_cp4))
print(np.min(diff))
print("\n")
print(np.max(img_cp0))
print(np.max(img_cp4))
print(np.max(diff))

fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(img_cp0)
ax[0, 0].set_title("before")
ax[0, 1].imshow(img_cp4)
ax[0, 1].set_title("after")
ax[1, 0].imshow(diff)
ax[1, 0].set_title("the difference")
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