import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import binary_opening, binary_closing, binary_erosion, binary_dilation
from scipy.signal import fftconvolve
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage.feature import canny
from copy import deepcopy

from functions_lars import get_image
from odemis.dataio import tiff
import cv2


class PreProcessingData:

    def __init__(self, image_before, image_after):
        self._og_image_before = image_before
        self._og_image_after = image_after
        self.image_before = None
        self.image_after = None
        self.image_before_blurred = None
        self.image_after_blurred = None
        self.rescaling_factor = None
        self.shift = None

    def rescaling(self):
        """
        To rescale a smaller image (k x l) to a bigger image (m x n).
        The dimension of the smaller image goes from (k x l) to (m x n).
        If they are of the same size, nothing happens and the input is returned.

        Parameters:
            img_before (ndarray):   The first image to compare.
            img_after (ndarray):    The second image to compare.

        Returns:
            img_before (ndarray):   If it was the smaller image, the rescaled image, otherwise the same as the input image.
            img_after (ndarray):    If it was the smaller image, the rescaled image, otherwise the same as the input image.
            scaling_factor (float): The ratio between the two image sizes: img_before/img_after.
        """

        if self.rescaling_factor is not None:
            self.image_before = self._og_image_before
            self.image_after = self._og_image_after
            self.image_before_blurred = None
            self.image_after_blurred = None
            self.rescaling_factor = None
            self.shift = None

        if self.image_before.shape == self.image_after.shape:
            self.rescaling_factor = 1
            return

        magni = self.image_before.shape[0] / self.image_after.shape[0]
        magni_x = self.image_before.shape[1] / self.image_after.shape[1]

        if magni != magni_x:
            raise ValueError(
                "Image shapes cannot be rescaled to one another. Distortion of the images would take place.")

        self.rescaling_factor = magni_x

        if magni < 1:
            magni = 1 / magni
            img_rescaled = np.zeros(self.image_after.shape)
            img_rescaled2 = np.zeros(self.image_after.shape)
            img_to_be_scaled = deepcopy(self.image_before)

        else:
            img_rescaled = np.zeros(self.image_before.shape)
            img_rescaled2 = np.zeros(self.image_before.shape)
            img_to_be_scaled = deepcopy(self.image_after)

        if magni % 1 == 0:
            for i in range(img_to_be_scaled.shape[0]):
                for j in range(int(magni)):
                    img_rescaled[int(magni * i + j), :img_to_be_scaled.shape[1]] = img_to_be_scaled[i, :]

            for i in range(img_to_be_scaled.shape[1]):
                for j in range(int(magni)):
                    img_rescaled2[:, int(magni * i + j)] = img_rescaled[:, i]

            if magni_x < 1:
                self.image_before = img_rescaled2
            else:
                self.image_after = img_rescaled2

        else:
            if magni_x < 1:
                img_to_be_scaled_too = np.array(self.image_after)
            else:
                img_to_be_scaled_too = np.array(self.image_before)

            extra_mag = np.arange(1, 10)
            possible = extra_mag * magni
            it = np.where(possible % 1 == 0)[0]

            if len(it) == 0:
                raise ValueError("Images are too difficult to rescale to one another, interpolation must be used "
                                 "(which is not implemented).")

            extra_mag = extra_mag[np.min(it)]
            first_rescaled = np.zeros((int(extra_mag * img_rescaled.shape[0]), int(extra_mag * img_rescaled.shape[1])))
            first_rescaled2 = np.zeros(first_rescaled.shape)
            second_rescaled = np.zeros(first_rescaled.shape)
            second_rescaled2 = np.zeros(first_rescaled.shape)

            for i in range(img_rescaled.shape[0]):
                for j in range(int(extra_mag)):
                    first_rescaled[int(extra_mag * i + j), :img_rescaled.shape[1]] = img_to_be_scaled_too[i, :]

            for i in range(img_rescaled.shape[1]):
                for j in range(int(extra_mag)):
                    first_rescaled2[:, int(extra_mag * i + j)] = first_rescaled[:, i]

            for i in range(img_to_be_scaled.shape[0]):
                for j in range(int(magni * extra_mag)):
                    second_rescaled[int(magni * extra_mag * i + j), :img_to_be_scaled.shape[1]] = img_to_be_scaled[i, :]

            for i in range(img_to_be_scaled.shape[1]):
                for j in range(int(magni * extra_mag)):
                    second_rescaled2[:, int(magni * extra_mag * i + j)] = second_rescaled[:, i]

            if magni_x < 1:
                self.image_before = second_rescaled2
                self.image_after = first_rescaled2
            else:
                self.image_before = first_rescaled2
                self.image_after = second_rescaled2

    def overlay(self, max_shift=5):
        """
        Shift the self.image_after in x and y position to have the best overlay with self.image_before.
        This is done with finding the maximum in a convolution.

        Parameters:
            self.image_before (ndarray):   The image which will be shifted to for the best overlay.
            self.image_after (ndarray):    The image which will shift to self.image_before for the best overlay.
            max_shift (int):               It is assumed that the images are roughly at the same position,
                                           so no large shifts are needed. On all sides of the convolution image,
                                           1/max_shift of the image is set to be zero. So with 1/8 you have a max shift
                                           of 3/8 in x and y direction (-3/8*img_shape, 3/8*img_shape). If you don't
                                           want a constraint on the shift, set max_shift to <=1.

        Returns:
            self.image_after (ndarray):    The shifted image, the values outside the boundaries of the image are set to
                                           the max value of the image so that in further processing steps those regions
                                           are not used.
            self.shift (ndarray):          The shift values in dy, dx
        """

        if self.shift is not None:
            # self.image_before, self.image_after = PreProcessingData.rescaling(self._og_image_before, self._og_image_after, params=logbook)
            self.image_before_blurred = None
            self.image_after_blurred = None
            self.shift = None


        # max shift is between 1 and inf, the higher, the higher the shift can be
        # if it is 1 no constraints will be given on the shift
        mean_before = np.mean(self.image_before)
        mean_after = np.mean(self.image_after)
        self.image_before = self.image_before - mean_before
        self.image_after = self.image_after - mean_after
        conv = fftconvolve(self.image_before, self.image_after[::-1, ::-1])

        # max_shift constraints
        if (int(conv.shape[0] / max_shift) > 0) & (max_shift > 1):
            conv[:int(conv.shape[0] / max_shift), :] = 0
            conv[-int(conv.shape[0] / max_shift):, :] = 0
            conv[:, :int(conv.shape[1] / max_shift)] = 0
            conv[:, -int(conv.shape[1] / max_shift):] = 0

        # calculating the shift in y and x with finding the maximum in the convolution
        shift = np.where(conv == np.max(conv))
        shift = np.asarray(shift)
        shift[0] = shift[0] - (conv.shape[0] - 1) / 2
        shift[1] = shift[1] - (conv.shape[1] - 1) / 2


        self.shift = shift
        self.image_before = self.image_before + mean_before
        self.image_after = self.image_after + mean_after

        dy_pix = int(shift[0])
        dx_pix = int(shift[1])

        print("dx is {} pixels".format(dx_pix))
        print("dy is {} pixels".format(dy_pix))

        # shifting the after milling image towards the before milling image to overlap nicely
        if dx_pix > 0:
            self.image_after[:, dx_pix:] = self.image_after[:, :-dx_pix]
            self.image_after[:, :dx_pix] = np.max(self.image_after)
        elif dx_pix < 0:
            self.image_after[:, :dx_pix] = self.image_after[:, -dx_pix:]
            self.image_after[:, dx_pix:] = np.max(self.image_after)
        if dy_pix > 0:
            self.image_after[dy_pix:, :] = self.image_after[:-dy_pix, :]
            self.image_after[:dy_pix, :] = np.max(self.image_after)
        elif dy_pix < 0:
            self.image_after[:dy_pix, :] = self.image_after[-dy_pix:, :]
            self.image_after[dy_pix:, :] = np.max(self.image_after)

    def blur_and_norm(self, blur=25):
        """
        Because the images before and after milling have different intensity profiles and do not have the exact same noise,
        normalization and blurring is required for being able to compare the two images. Here the image is first
        blurred with a Gaussian filter and afterwards normalized to a range from 0 to 1. The unblurred image is also
        normalized with its own maximum and the minimum of the blurred images. The minimum of the blurred image is used
        because everything below this is probably noise.

        Parameters:
            img (ndarray):              The image to be blurred and normalized.
            blur (int):                 The sigma for the gaussian blurring, the higher the number, the more blurring occurs.

        Returns:
            img (ndarray):              The unblurred, normalized image before milling.
            img_blurred (ndarray):      The blurred, normalized image before milling.
        """
        if self._image_before_blurred is not None:
            print("This function was already performed.")
            return

        # blurring the images
        self._image_before_blurred = gaussian_filter(self.image_before, sigma=blur)
        self._image_after_blurred = gaussian_filter(self.image_after, sigma=blur)

        # normalization of the blurred images to [0,1] interval
        base_lvl_before = np.min(self._image_before_blurred)
        base_lvl_after = np.min(self._image_after_blurred)
        self._image_before_blurred = (self._image_before_blurred - base_lvl_before) / (
                np.max(self._image_before_blurred) - base_lvl_before)
        self._image_after_blurred = (self._image_after_blurred - base_lvl_after) / (
                np.max(self._image_after_blurred) - base_lvl_after)

        # normalization of the initial images to [0,1] interval
        # you take base_lvl of the blurred images to be the minimum intensity value because lower intensity values are
        # probably due to noise.
        self.image_before = (self.image_before - base_lvl_before) / (np.max(self.image_before) - base_lvl_before)
        self.image_before[self.image_before < 0] = 0
        self.image_after = (self.image_after - base_lvl_after) / (np.max(self.image_after) - base_lvl_after)
        self.image_after[self.image_after < 0] = 0

        return self._image_before_blurred, self._image_after_blurred


# Roi_detector = Automatic_ROI_Detection(from_path(image_path), param2, param3)

data_paths_before = ["/home/victoria/Documents/Lars/data/1/FOV2_GFP_cp00.tif",
                     "/home/victoria/Documents/Lars/data/2/FOV1_checkpoint_00.tiff",
                     "/home/victoria/Documents/Lars/data/3/FOV4_checkpoint_00.tiff",
                     "/home/victoria/Documents/Lars/data/4/METEOR_images/FOV3_checkpoint_00.tif",
                     "/home/victoria/Documents/Lars/data/6/20201002_FOV2_checkpoint_001_stack_001.tiff",
                     "/home/victoria/Documents/Lars/data/7/FOV3_Meteor_stacks/Lamella_slices_tif"
                     "/EA010_8_FOV3_checkpoint_00_1-24.tif",
                     "/home/victoria/Documents/Lars/data/meteor1/FOV3_Meteor_stacks/EA010_8_FOV3_checkpoint_00_1.tiff",
                     "/home/victoria/Documents/Lars/data/XA Yeast-20220215T093027Z-001/XA "
                     "Yeast/20200918_millingsite_start.ome.tiff",
                     "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Mammalian_cells/FOV7_cp00_GFP.tif",
                     "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Mammalian_cells/FOV9_cp00_GFP.tif",
                     "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Mammalian_cells/FOV11_cp00_GFP.tif",
                     "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Mammalian_cells/FOV12_cp00_GFP.tif",
                     "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Yeast_eGFP-Ede1/G2_FOV1_checkpoint00.tif",
                     "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Yeast_eGFP-Ede1/G2_FOV2_checkpoint00.tif",
                     "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Yeast_eGFP-Ede1/G2_FOV3_checkpoint00.tif",
                     "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Yeast_eGFP-Ede1/G2_FOV4_checkpoint00.tif",
                     "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/negative_examples/yeast/"
                     "FOV3_new_checkpoint_00.tiff",
                     "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/negative_examples/yeast/"
                     "FOV6_checkpoint00_ch00.tiff"
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
                    "Yeast/20200918-zstack_400nm.ome.tiff",
                    "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Mammalian_cells/FOV7_after_GFP.tif",
                    "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Mammalian_cells/FOV9_after_GFP.tif",
                    "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Mammalian_cells/FOV11_after_GFP.tif",
                    "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Mammalian_cells/FOV12_after_GFP.tif",
                    "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Yeast_eGFP-Ede1/G2_FOV1_checkpoint03.tif",
                    "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Yeast_eGFP-Ede1/G2_FOV2_checkpoint04.tif",
                    "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Yeast_eGFP-Ede1/G2_FOV3_checkpoint04.tif",
                    "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/Yeast_eGFP-Ede1/G2_FOV4_checkpoint04.tif",
                    "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/negative_examples/yeast/"
                    "FOV3_new_final_lamella.tiff",
                    "/home/victoria/Documents/Lars/data/Meteor_data_for_Lars/negative_examples/yeast/"
                    "FOV6_checkpoint02_ch00.tiff"
                    ]

ll = len(data_paths_before)
channel_before = np.zeros(ll, dtype=int)
channel_before[7] = 1
channel_after = np.zeros(ll, dtype=int)

img_before, img_after, meta_before, meta_after = get_image(data_paths_before[0], data_paths_after[0],
                                                           channel_before[0], channel_after[0],
                                                           mode='projection', proj_mode='max')

data = PreProcessingData(img_before, img_after)
print(data)

data.rescaling()
data.overlay()
data.blur_and_norm()
