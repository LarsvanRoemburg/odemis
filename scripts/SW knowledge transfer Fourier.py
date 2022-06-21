import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import rotate
from scipy.signal import fftconvolve
import cv2

xrange = np.arange(-50, 51, 1)
img = np.sin(0.5*xrange)*np.ones((101, 101))
fft_img = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img)))

fig, ax = plt.subplots(ncols=2)
ax[0].imshow(img)
ax[1].imshow(fft_img.real ** 2 + fft_img.imag ** 2)

fft_img = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img.T)))
fig, ax = plt.subplots(ncols=2)
ax[0].imshow(img.T)
ax[1].imshow(fft_img.real ** 2 + fft_img.imag ** 2)

img_r = rotate(img, 45)
fft_img = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img_r)))
fig, ax = plt.subplots(ncols=2)
ax[0].imshow(img_r)
ax[1].imshow(fft_img.real ** 2 + fft_img.imag ** 2)
plt.show()

image = cv2.imread("/home/victoria/Documents/Lars/noisy image.png", 0)

fft_image = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(image)))

fig, ax = plt.subplots(ncols=2)
ax[0].imshow(image)
ax[1].imshow(np.log(fft_image.real ** 2 + fft_image.imag ** 2))
plt.show()

for a in [10, 6, 3, 2.5]:
    blurred_fft = np.array(fft_image)
    blurred_fft[:int(fft_image.shape[0]/a), :] = 0
    blurred_fft[-int(fft_image.shape[0]/a):, :] = 0
    blurred_fft[:, :int(fft_image.shape[1]/a)] = 0
    blurred_fft[:, -int(fft_image.shape[1]/a):] = 0

    blurred_img = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(blurred_fft)))

    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(blurred_img.real)
    ax[1].imshow(np.log(blurred_fft.real ** 2 + blurred_fft.imag ** 2))
    plt.show()

for b in [5, 15, 30]:
    blurred_fft = np.array(fft_image)
    blurred_fft[int(fft_image.shape[0]/2-b):int(fft_image.shape[0]/2+b), int(fft_image.shape[0]/2-b):int(fft_image.shape[0]/2+b)] = 0

    blurred_img = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(blurred_fft)))

    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(blurred_img.real)
    ax[1].imshow(np.log(blurred_fft.real ** 2 + blurred_fft.imag ** 2))
    plt.show()

kernel = np.zeros((11, 11))
kernel[4:7, 4:7] = 1
fft_kernel = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(kernel)))

kernel = np.zeros((11, 11))
kernel[4:7, 4:7] = 1
kernel2 = np.zeros(image.shape)
ymid = int(kernel2.shape[0]/2)
xmid = int(kernel2.shape[1]/2)
ylen = int(kernel.shape[0]/2)
xlen = int(kernel.shape[1]/2)
kernel2[ymid-ylen:ymid+ylen+1, xmid-xlen:xmid+xlen+1] = kernel
fft_kernel = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(kernel2)))

fig, ax = plt.subplots(ncols=3)
ax[0].imshow(kernel)
ax[1].imshow(kernel2)
ax[2].imshow(np.log(fft_kernel.real**2+fft_kernel.imag**2))
plt.show()

fft_combined = fft_image * fft_kernel

filtered_img = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fft_combined)))

fig, ax = plt.subplots(ncols=2)
ax[0].imshow(np.log(fft_combined.real**2 + fft_combined.imag**2))
ax[1].imshow(filtered_img.real)
plt.show()

new_image = fftconvolve(image, kernel)
new_image = new_image[int((kernel.shape[0]-1)/2):-int((kernel.shape[0]-1)/2), int((kernel.shape[1]-1)/2):-int((kernel.shape[1]-1)/2)]
fig, ax = plt.subplots(ncols=3, sharey=True, sharex=True)
ax[0].imshow(image)
ax[1].imshow(filtered_img.real)
ax[2].imshow(new_image)
plt.show()

blurred_img = gaussian_filter(image, 1)
xrange = np.arange(0, 256, 1)
img = np.mean(blurred_img)*np.sin(0.5*xrange)*np.ones((256, 256))
lines_in_img = blurred_img + img

fig, ax = plt.subplots(ncols=3)
ax[0].imshow(blurred_img)
ax[1].imshow(img)
ax[2].imshow(lines_in_img)
plt.show()

fft_image_and_lines = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(lines_in_img)))
fft_lines = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img)))

fig, ax = plt.subplots(ncols=3, nrows=2)
ax[0, 0].imshow(blurred_img)
ax[0, 1].imshow(img)
ax[0, 2].imshow(lines_in_img)
ax[1, 0].imshow(np.log(fft_image.real**2 + fft_image.imag**2))
ax[1, 1].imshow(np.log(fft_lines.real**2 + fft_lines.imag**2))
ax[1, 2].imshow(np.log(fft_image_and_lines.real**2 + fft_image_and_lines.imag**2))
plt.show()

fft_image_without = fft_image_and_lines - fft_lines
img_without = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fft_image_without)))

fig, ax = plt.subplots(ncols=3, nrows=2)
ax[0, 0].imshow(lines_in_img)
ax[0, 1].imshow(img)
ax[0, 2].imshow(img_without.real)
ax[1, 0].imshow(np.log(fft_image_and_lines.real**2 + fft_image_and_lines.imag**2))
ax[1, 1].imshow(np.log(fft_lines.real**2 + fft_lines.imag**2))
ax[1, 2].imshow(np.log(fft_image_without.real**2 + fft_image_without.imag**2))
plt.show()
