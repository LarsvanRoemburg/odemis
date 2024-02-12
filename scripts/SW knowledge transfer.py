import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import canny
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import fftconvolve
import cv2
from odemis.dataio import tiff


image = cv2.imread("/home/victoria/Documents/Lars/19-aug_cancer-fatty-jackets.jpg", 0)
fig, ax = plt.subplots(ncols=2, nrows=2)
ax[0, 0].imshow(image)
ax[0, 0].set_title("Input image")

Gx = fftconvolve(image, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
ax[0, 1].imshow(Gx)
ax[0, 1].set_title("Gx")

Gy = fftconvolve(image, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))
ax[1, 0].imshow(Gy)
ax[1, 0].set_title("Gy")

G = np.sqrt(Gx**2 + Gy**2)
Theta = np.arctan(Gy/Gx)
ax[1, 1].imshow(G)
ax[1, 1].set_title("G")

plt.show()

# # image2 = tiff.read_data("/home/victoria/Documents/Lars/data/XA Yeast-20220215T093027Z-001/XA Yeast/20200918-zstack_400nm.ome.tiff")
# # image2 = image2[0][0][0][3]

image2 = cv2.imread("/home/victoria/Documents/Lars/noisy image.png", 0)
fig, ax = plt.subplots(ncols=2, nrows=2)
ax[0, 0].imshow(image2)
ax[0, 0].set_title("Input image")

Gx = fftconvolve(image2, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
ax[0, 1].imshow(Gx)
ax[0, 1].set_title("Gx")

Gy = fftconvolve(image2, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))
ax[1, 0].imshow(Gy)
ax[1, 0].set_title("Gy")

G = np.sqrt(Gx**2 + Gy**2)
ax[1, 1].imshow(G)
ax[1, 1].set_title("G")

plt.show()

image2 = gaussian_filter(image2, sigma=2)
fig, ax = plt.subplots(ncols=2, nrows=2)
ax[0, 0].imshow(image2)
ax[0, 0].set_title("Input image")

Gx = fftconvolve(image2, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
ax[0, 1].imshow(Gx)
ax[0, 1].set_title("Gx")

Gy = fftconvolve(image2, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))
ax[1, 0].imshow(Gy)
ax[1, 0].set_title("Gy")

G = np.sqrt(Gx**2 + Gy**2)
ax[1, 1].imshow(G)
ax[1, 1].set_title("G")

plt.show()

edges = canny(image2, low_threshold=50, high_threshold=50)
thin_G = edges*G[1:-1, 1:-1]
thin_G[0, 0] = np.max(G)
fig, ax = plt.subplots(ncols=2)
ax[0].imshow(G)
ax[1].imshow(thin_G)
plt.show()

edges = np.array(edges*255, dtype=np.uint8)
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=30, minLineLength=2, maxLineGap=2)

lines_image = np.zeros(edges.shape)
print("{} lines detected.".format(len(lines)))
for points in lines:
    # Extracted points nested in the list
    x1, y1, x2, y2 = points[0]
    # Draw the lines joining the points on the original image
    cv2.line(lines_image, (x1, y1), (x2, y2), 255, 1)

fig, ax = plt.subplots(ncols=2)
ax[0].imshow(edges)
ax[1].imshow(lines_image)
plt.show()
