import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import fftconvolve
import cv2

image = cv2.imread("/home/victoria/Documents/Lars/noisy image.png", 0)

fft_image = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(image)))

# fig, ax = plt.subplots(ncols=2)
# ax[0].imshow(image)
# ax[1].imshow(np.log(fft_image.real ** 2 + fft_image.imag ** 2))
# plt.show()
#
# a = 10
# blurred_fft = np.array(fft_image)
# blurred_fft[:int(fft_image.shape[0]/a), :] = 0
# blurred_fft[-int(fft_image.shape[0]/a):, :] = 0
# blurred_fft[:, :int(fft_image.shape[1]/a)] = 0
# blurred_fft[:, -int(fft_image.shape[1]/a):] = 0
#
# blurred_img = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(blurred_fft)))
#
# fig, ax = plt.subplots(ncols=2)
# ax[0].imshow(blurred_img.real)
# ax[1].imshow(np.log(blurred_fft.real ** 2 + blurred_fft.imag ** 2))
# plt.show()
#
# a = 6
# blurred_fft = np.array(fft_image)
# blurred_fft[:int(fft_image.shape[0]/a), :] = 0
# blurred_fft[-int(fft_image.shape[0]/a):, :] = 0
# blurred_fft[:, :int(fft_image.shape[1]/a)] = 0
# blurred_fft[:, -int(fft_image.shape[1]/a):] = 0
#
# blurred_img = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(blurred_fft)))
#
# fig, ax = plt.subplots(ncols=2)
# ax[0].imshow(blurred_img.real)
# ax[1].imshow(np.log(blurred_fft.real ** 2 + blurred_fft.imag ** 2))
# plt.show()
#
# a = 3
# blurred_fft = np.array(fft_image)
# blurred_fft[:int(fft_image.shape[0]/a), :] = 0
# blurred_fft[-int(fft_image.shape[0]/a):, :] = 0
# blurred_fft[:, :int(fft_image.shape[1]/a)] = 0
# blurred_fft[:, -int(fft_image.shape[1]/a):] = 0
#
# blurred_img = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(blurred_fft)))
#
# fig, ax = plt.subplots(ncols=2)
# ax[0].imshow(blurred_img.real)
# ax[1].imshow(np.log(blurred_fft.real ** 2 + blurred_fft.imag ** 2))
# plt.show()
#
# a = 2.5
# blurred_fft = np.array(fft_image)
# blurred_fft[:int(fft_image.shape[0]/a), :] = 0
# blurred_fft[-int(fft_image.shape[0]/a):, :] = 0
# blurred_fft[:, :int(fft_image.shape[1]/a)] = 0
# blurred_fft[:, -int(fft_image.shape[1]/a):] = 0
#
# blurred_img = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(blurred_fft)))
#
# fig, ax = plt.subplots(ncols=2)
# ax[0].imshow(blurred_img.real)
# ax[1].imshow(np.log(blurred_fft.real ** 2 + blurred_fft.imag ** 2))
# plt.show()
#
# b = 5
# blurred_fft = np.array(fft_image)
# blurred_fft[int(fft_image.shape[0]/2-b):int(fft_image.shape[0]/2+b), int(fft_image.shape[0]/2-b):int(fft_image.shape[0]/2+b)] = 0
#
# blurred_img = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(blurred_fft)))
#
# fig, ax = plt.subplots(ncols=2)
# ax[0].imshow(blurred_img.real)
# ax[1].imshow(np.log(blurred_fft.real ** 2 + blurred_fft.imag ** 2))
# plt.show()
#
# b = 15
# blurred_fft = np.array(fft_image)
# blurred_fft[int(fft_image.shape[0]/2-b):int(fft_image.shape[0]/2+b), int(fft_image.shape[0]/2-b):int(fft_image.shape[0]/2+b)] = 0
#
# blurred_img = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(blurred_fft)))
#
# fig, ax = plt.subplots(ncols=2)
# ax[0].imshow(blurred_img.real)
# ax[1].imshow(np.log(blurred_fft.real ** 2 + blurred_fft.imag ** 2))
# plt.show()
#
# b = 30
# blurred_fft = np.array(fft_image)
# blurred_fft[int(fft_image.shape[0]/2-b):int(fft_image.shape[0]/2+b), int(fft_image.shape[0]/2-b):int(fft_image.shape[0]/2+b)] = 0
#
# blurred_img = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(blurred_fft)))
#
# fig, ax = plt.subplots(ncols=2)
# ax[0].imshow(blurred_img.real)
# ax[1].imshow(np.log(blurred_fft.real ** 2 + blurred_fft.imag ** 2))
# plt.show()

kernel = np.zeros((11, 11))
kernel[4:7, 4:7] = 1
fft_kernel = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(kernel)))

kernel = np.zeros((101, 101))
kernel[48:53, 48:53] = 1
fft_kernel = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(kernel)))

fig, ax = plt.subplots(ncols=2)
ax[0].imshow(kernel)
ax[1].imshow(np.log(fft_kernel.real**2+fft_kernel.imag**2))
plt.show()

fft_kernel2 = np.zeros(fft_image.shape, dtype=complex)

ymid = int(fft_kernel2.shape[0]/2)
xmid = int(fft_kernel2.shape[1]/2)
ylen = int(fft_kernel.shape[0]/2)
xlen = int(fft_kernel.shape[1]/2)

fft_kernel2[ymid-ylen:ymid+ylen+1, xmid-xlen:xmid+xlen+1] = fft_kernel
fft_combined = fft_image * fft_kernel2

filtered_img = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fft_combined)))

fig, ax = plt.subplots(ncols=2)
ax[0].imshow(np.log(fft_combined.real**2 + fft_combined.imag**2))
ax[1].imshow(filtered_img.real)
plt.show()
# plt.imshow(filtered_img)
# plt.imshow(filtered_img.real)
new_image = fftconvolve(image, kernel)
new_image = new_image[int((kernel.shape[0]-1)/2):-int((kernel.shape[0]-1)/2), int((kernel.shape[1]-1)/2):-int((kernel.shape[1]-1)/2)]
fig, ax = plt.subplots(ncols=3)
ax[0].imshow(image)
ax[1].imshow(filtered_img.real)
ax[2].imshow(new_image)
plt.show()
