import logging
import unittest
import gc

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from odemis.dataio import tiff
from scripts.functions_lars import find_focus_z_slice_and_outlier_cutoff, rescaling, overlay, \
    z_projection_and_outlier_cutoff, blur_and_norm, create_diff_mask, find_x_or_y_pos_milling_site, calculate_angles, \
    create_line_mask, combine_masks, create_masked_img, create_binary_end_image, get_image, find_lines, \
    combine_and_constraint_lines, group_single_lines, couple_groups_of_lines, detect_blobs, combine_groups, \
    combine_biggest_groups, last_group_selection
from try_out_lars import data_paths_before, data_paths_after, data_paths_true_masks
from copy import deepcopy


class TestFunctionsLars(unittest.TestCase):
    """My test cases testing my tests"""

    ch_before = None
    correct_scaling = None

    @classmethod
    def setUpClass(cls):
        cls.path_before = data_paths_before
        cls.path_after = data_paths_after
        cls.path_masks = data_paths_true_masks

        ll = len(data_paths_before)
        cls.ch_before = np.zeros(ll, dtype=int)
        cls.ch_before[7] = int(1)
        cls.ch_after = np.zeros(ll, dtype=int)

        # found manually after scaling and overlay
        cls.correct_milling_pos_y = [558, 945, 1022, 1182, 1040, 1368, 1355, 1050, 430, 470, 490, 630, 1035, 640, 850,
                                     1114, 1025, 800]
        cls.correct_milling_pos_x = [918, 1291, 1673, 1458, 1535, 1355, 1355, 1144, 610, 930, 820, 630, 1140, 1800,
                                     1955, 950, 1035, 1850]

        cls.correct_z_slice_before = [19, 23, 23, 26, 33, 0, 23, 0, 27, 22, 22, 23, 26, 36, 23, 25, 24, 24]
        cls.correct_z_slice_after = [10, 15, 22, 19, 32, 0, 17, 2, 22, 18, 25, 22, 17, 15, 16, 17, 23, 24]

        cls.correct_shift = np.zeros((18, 2))

        # img_before size / img_after size
        cls.correct_scaling = np.ones(18, dtype=float)
        cls.correct_scaling[2] = float(2)
        cls.correct_scaling[17] = float(0.5)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    # def test_my_function(self):
    #     """My test function testing my functions"""
    #     expected = 2
    #     result = my_function(1)
    #     self.assertEqual(expected, result)

    def test_focus_z_slice(self):
        for i in np.array([0, 5, 12]):
            data = tiff.read_data(data_paths_before[i])

            if self.correct_scaling[i] >= 1:
                img, result = find_focus_z_slice_and_outlier_cutoff(data,
                                                                    milling_pos_y=self.correct_milling_pos_y[i],
                                                                    milling_pos_x=self.correct_milling_pos_x[i],
                                                                    which_channel=self.ch_before[i],
                                                                    square_width=1 / 4,
                                                                    num_slices=5, outlier_cutoff=99, mode='max')

            else:
                img, result = find_focus_z_slice_and_outlier_cutoff(data,
                                                                    milling_pos_y=self.correct_milling_pos_y[i] *
                                                                                  self.correct_scaling[i],
                                                                    milling_pos_x=self.correct_milling_pos_x[i] *
                                                                                  self.correct_scaling[i],
                                                                    which_channel=self.ch_before[i],
                                                                    square_width=1 / 4,
                                                                    num_slices=5, outlier_cutoff=99, mode='max')

            correct = (np.abs(result - self.correct_z_slice_before[i]) <= 2)
            # np.array_equal
            self.assertTrue(correct)
            del data, img, result
            gc.collect()

    def test_z_projection(self):
        for i in np.array([0, 5, 12]):
            data = tiff.read_data(data_paths_before[i])

            img = z_projection_and_outlier_cutoff(data, 30, which_channel=self.ch_before[i], outlier_cutoff=99,
                                                  mode='max')

            if data[0].shape[0] == 1:
                s = int((len(data[self.ch_before[i]][0][0]) - 30) / 2)
                img_correct = np.array(data[self.ch_before[i]][0][0][s:-s], dtype=int)
                histo, bins = np.histogram(img_correct, bins=2000)
                histo = np.cumsum(histo)
                histo = histo / histo[-1] * 100
                plc = np.min(np.where(histo > 99))
                thr_out = bins[plc] / 2 + bins[plc + 1] / 2
                img_correct[img_correct > thr_out] = thr_out

                img_correct = np.max(img_correct, axis=0)

            else:
                img_correct = np.array(data[self.ch_before[i]], dtype=int)
                histo, bins = np.histogram(img_correct, bins=2000)
                histo = np.cumsum(histo)
                histo = histo / histo[-1] * 100
                plc = np.min(np.where(histo > 99))
                thr_out = bins[plc] / 2 + bins[plc + 1] / 2
                img_correct[img_correct > thr_out] = thr_out

            self.assertTrue(np.array_equal(img, img_correct))
            # use np.testing.assert_almost_equal() np.testing.assert_allclose() np.testing.assert_array_almost_equal()

    def test_rescaling(self):
        n = 20
        m = 10
        a = np.zeros((n, n))
        b = np.ones((m, m))
        b[0, :] = np.arange(m)

        result_a, result_b, result_scaling = rescaling(a, b)

        correct_a = np.zeros((n, n))
        correct_b = np.ones((n, n))
        correct_b[0:2, :] = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9])

        correct_scaling = float(n / m)

        self.assertTrue(np.array_equal(result_a, correct_a))
        self.assertTrue(np.array_equal(result_b, correct_b))
        self.assertEqual(result_scaling, correct_scaling)

        n = 5
        m = 8
        a = np.zeros((n, n))
        a[0, :] = 1  # np.arange(n)
        b = np.ones((m, m))

        result_a, result_b, result_scaling = rescaling(a, b)

        correct_a = np.zeros((n * m, n * m))
        correct_a[0:m, :] = 1
        correct_b = np.ones((n * m, n * m))
        correct_scaling = float(n / m)

        self.assertTrue(np.array_equal(result_a, correct_a))
        self.assertTrue(np.array_equal(result_b, correct_b))
        self.assertEqual(result_scaling, correct_scaling)

        n = 10
        m = 20
        a = np.zeros((n, n))
        a[0, :] = np.arange(n)
        b = np.ones((m, m))

        result_a, result_b, result_scaling = rescaling(a, b)

        correct_a = np.zeros((m, m))
        correct_a[0:2, :] = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9])
        correct_b = np.ones((m, m))
        correct_scaling = float(n / m)

        self.assertTrue(np.array_equal(result_a, correct_a))
        self.assertTrue(np.array_equal(result_b, correct_b))
        self.assertEqual(result_scaling, correct_scaling)

        n = 10
        m = 30
        a = np.zeros((n, n))
        a[0, :] = np.arange(n)
        b = np.ones((m, m))

        result_a, result_b, result_scaling = rescaling(a, b)

        correct_a = np.zeros((m, m))
        correct_a[0:3, :] = np.array(
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9])
        correct_b = np.ones((m, m))
        correct_scaling = float(n / m)

        self.assertTrue(np.array_equal(result_a, correct_a))
        self.assertTrue(np.array_equal(result_b, correct_b))
        self.assertEqual(result_scaling, correct_scaling)

        n = 20
        m = 30
        a = np.zeros((n, n))
        a[0, :] = np.arange(n)
        b = np.ones((m, m))

        result_a, result_b, result_scaling = rescaling(a, b)

        correct_a = np.zeros((2 * m, 2 * m))
        d = np.zeros(2 * m)
        for i in range(len(d)):
            d[i] = int(i // 3)

        correct_a[0:3, :] = d
        correct_b = np.ones((2 * m, 2 * m))
        correct_scaling = float(n / m)

        self.assertTrue(np.array_equal(result_a, correct_a))
        self.assertTrue(np.array_equal(result_b, correct_b))
        self.assertEqual(result_scaling, correct_scaling)

        n = 2
        m = 50

        a = np.zeros((n, n))
        a[0, :] = 1
        b = np.ones((m, m))

        result_a, result_b, result_scaling = rescaling(a, b)

        correct_a = np.zeros((m, m))
        correct_a[:int(correct_a.shape[0] / 2), :] = 1
        correct_b = np.ones((m, m))
        correct_scaling = float(n / m)

        self.assertTrue(np.array_equal(result_a, correct_a))
        self.assertTrue(np.array_equal(result_b, correct_b))
        self.assertEqual(result_scaling, correct_scaling)

        n = 37
        m = 41

        a = np.zeros((n, n))
        b = np.ones((m, m))

        with self.assertRaises(ValueError):
            rescaling(a, b)

    def test_overlay(self):
        n = 101
        for i in range(11):
            a = np.zeros((n, n))
            b = np.zeros((n, n))
            a[50, 50] = 1
            b[int(i * 10), int(i * 10)] = 1

            a_result, b_result, shift = overlay(a, b, 1)
            correct_shift = np.array([[50 - int(i * 10)], [50 - int(i * 10)]])

            self.assertTrue(np.array_equal(shift, correct_shift))

        for i in range(11):
            a = np.zeros((n, n))
            b = np.zeros((n, n))
            a[50, 50] = 1
            b[50, int(i * 10)] = 1

            a_result, b_result, shift = overlay(a, b, 1)
            correct_shift = np.array([[0], [50 - int(i * 10)]])

            self.assertTrue(np.array_equal(shift, correct_shift))

        for i in range(11):
            a = np.zeros((n, n))
            b = np.zeros((n, n))
            a[50, 50] = 1
            b[int(i * 10), 50] = 1

            a_result, b_result, shift = overlay(a, b, 1)
            correct_shift = np.array([[50 - int(i * 10)], [0]])

            self.assertTrue(np.array_equal(shift, correct_shift))

        for i in range(10):
            n = 200
            a = np.zeros((n, n))
            b = np.zeros((n, n))
            a[int(n / 2), int(n / 2)] = 1
            b[int(i * n / 10), int(n / 2)] = 1

            a_result, b_result, shift = overlay(a, b, 4)
            if i < 3 or i > 7:
                correct_shift = np.array([[0], [0]])
            else:
                correct_shift = np.array([[int(n / 2 - i * n / 10)], [0]])

            self.assertTrue(np.array_equal(shift, correct_shift))

        n = 101
        a = np.zeros((n, n))

        a[50, 50] = 1
        for j in range(7):
            b = np.zeros((n, n))
            for i in range(3):
                b[int(i * 10 + (j + 1) * 10), 50] = 1

            a_result, b_result, shift = overlay(a, b, 1)
            t = np.array([10, 20, 30])
            e = np.min(np.abs(t + j * 10 - 50))
            if j > 4:
                e = -e
            correct_shift = np.array([[e], [0]])

            self.assertTrue(np.array_equal(shift, correct_shift))

    def test_get_image(self):
        """
        Here I do still use the functions z_projection_and_outlier_cutoff() and find_focus_z_slice_and_outlier_cutoff().
        """

        for i in np.array([0, 5]):
            img_before, img_after, meta_before, meta_after = get_image(self.path_before[i], self.path_after[i],
                                                                       self.ch_before[i], self.ch_after[i],
                                                                       max_slices_proj=30,
                                                                       max_slices_focus=5, mode='projection',
                                                                       proj_mode='max', blur=25)

            data_before_milling = tiff.read_data(self.path_before[i])
            # meta_before = data_before_milling[self.ch_before[i]].metadata
            img_before_correct = z_projection_and_outlier_cutoff(data_before_milling, 30, self.ch_before[i],
                                                                 mode='max')
            del data_before_milling
            gc.collect()

            data_after_milling = tiff.read_data(self.path_after[i])
            # meta_after = data_after_milling[self.ch_after[i]].metadata
            img_after_correct = z_projection_and_outlier_cutoff(data_after_milling, 30, self.ch_after[i],
                                                                mode='max')

            del data_after_milling
            gc.collect()

            self.assertTrue(np.array_equal(img_before, img_before_correct))
            self.assertTrue(np.array_equal(img_after, img_after_correct))

            img_before, img_after, meta_before, meta_after = get_image(self.path_before[i], self.path_after[i],
                                                                       self.ch_before[i], self.ch_after[i],
                                                                       max_slices_proj=30,
                                                                       max_slices_focus=5, mode='in_focus',
                                                                       proj_mode='mean', blur=25)

            data_before_milling = tiff.read_data(self.path_before[i])
            # meta_before = data_before_milling[ch_before].metadata
            data_after_milling = tiff.read_data(self.path_after[i])
            # meta_after = data_after_milling[ch_after].metadata

            # if both data sets have only one slice, just output those.
            if data_before_milling[0].shape[0] != 1 and data_after_milling[0].shape[0] != 1:
                img_before_correct = z_projection_and_outlier_cutoff(data_before_milling, 30, self.ch_before[i],
                                                                     mode='mean')
                img_after_correct = z_projection_and_outlier_cutoff(data_after_milling, 30, self.ch_after[i],
                                                                    mode='mean')
            else:
                # creating projection images for finding a rough estimate for milling site position
                img_before_correct = z_projection_and_outlier_cutoff(data_before_milling, 30, self.ch_before[i],
                                                                     mode='mean')
                img_after_correct = z_projection_and_outlier_cutoff(data_after_milling, 30, self.ch_after[i],
                                                                    mode='mean')

                # rescaling one image to the other if necessary
                img_before_correct, img_after_correct, magni = rescaling(img_before_correct, img_after_correct)
                # calculate the shift between the two images
                img_after_correct, shift = overlay(img_before_correct, img_after_correct, max_shift=4)
                # shift is [dy, dx]

                # blurring the images
                img_before_correct, img_before_blurred = blur_and_norm(img_before_correct, blur=25)
                img_after_correct, img_after_blurred = blur_and_norm(img_after_correct, blur=25)

                # finding the estimates for the milling site position
                milling_x_pos = find_x_or_y_pos_milling_site(img_before_blurred, img_after_blurred, ax='x')
                milling_y_pos = find_x_or_y_pos_milling_site(img_before_blurred, img_after_blurred, ax='y')

                # finding the in focus slice.
                # because the raw data is not yet scaled properly, the position needs to be adjusted for that
                if magni == 1:
                    img_before_correct, in_focus = find_focus_z_slice_and_outlier_cutoff(data_before_milling,
                                                                                         milling_y_pos,
                                                                                         milling_x_pos,
                                                                                         self.ch_before[i],
                                                                                         num_slices=5, mode='mean')
                    img_after_correct, in_focus = find_focus_z_slice_and_outlier_cutoff(data_after_milling,
                                                                                        milling_y_pos - shift[0],
                                                                                        milling_x_pos - shift[1],
                                                                                        self.ch_after[i],
                                                                                        num_slices=5, mode='mean')
                elif magni > 1:
                    img_before_correct, in_focus = find_focus_z_slice_and_outlier_cutoff(data_before_milling,
                                                                                         milling_y_pos,
                                                                                         milling_x_pos,
                                                                                         self.ch_before[i],
                                                                                         num_slices=5, mode='mean')
                    img_after_correct, in_focus = find_focus_z_slice_and_outlier_cutoff(data_after_milling,
                                                                                        (milling_y_pos - shift[
                                                                                            0]) / magni,
                                                                                        (milling_x_pos - shift[
                                                                                            1]) / magni,
                                                                                        self.ch_after[i],
                                                                                        num_slices=5, mode='mean')
                elif magni < 1:
                    img_before_correct, in_focus = find_focus_z_slice_and_outlier_cutoff(data_before_milling,
                                                                                         milling_y_pos * magni,
                                                                                         milling_x_pos * magni,
                                                                                         self.ch_before[i],
                                                                                         num_slices=5, mode='mean')
                    img_after_correct, in_focus = find_focus_z_slice_and_outlier_cutoff(data_after_milling,
                                                                                        milling_y_pos - shift[0],
                                                                                        milling_x_pos - shift[1],
                                                                                        self.ch_after[i],
                                                                                        num_slices=5, mode='mean')
            del data_before_milling, data_after_milling
            gc.collect()

            self.assertTrue(np.array_equal(img_before, img_before_correct))
            self.assertTrue(np.array_equal(img_after, img_after_correct))

    def test_blur_and_norm(self):
        img = np.zeros((100, 100))
        img[49, 49] = 1

        result, result_blurred = blur_and_norm(img, 1)

        correct = img
        correct_blurred = gaussian_filter(img, 1)
        correct_blurred = correct_blurred / np.max(correct_blurred)

        self.assertTrue(np.array_equal(result, correct))
        self.assertTrue(np.array_equal(result_blurred, correct_blurred))

        img = np.ones((100, 100))
        img[49, 49] = 2

        result, result_blurred = blur_and_norm(img, 5)

        correct = np.zeros((100, 100))
        correct[49, 49] = 1
        correct_blurred = gaussian_filter(img, 5) - 1
        correct_blurred = correct_blurred / np.max(correct_blurred)

        self.assertTrue(np.array_equal(result, correct))
        self.assertTrue(np.array_equal(result_blurred, correct_blurred))

    def test_create_diff_mask(self):
        n = 100
        img1 = np.ones((n, n))
        img2 = np.ones((n, n))

        img1[39:59, 39:59] = 2

        result = create_diff_mask(img1, img2, open_iter=2)
        correct = np.zeros((n, n), dtype=bool)
        correct[39:59, 39:59] = True

        self.assertTrue(np.array_equal(result, correct))

        result = create_diff_mask(img1, img2, open_iter=10)
        correct[39:59, 39:59] = False
        self.assertTrue(np.array_equal(result, correct))

        result = create_diff_mask(img1, img2, threshold_mask=0.6, open_iter=2)
        self.assertTrue(np.array_equal(result, correct))

        correct[41:57, 41:57] = True
        result = create_diff_mask(img1, img2, open_iter=2, ero_iter=2)
        self.assertTrue(np.array_equal(result, correct))

        img1[33:37, 33:37] = 2
        img1[20:25, 40:45] = 2
        img1[65:69, 51:55] = 2
        img1[40:45, 80:85] = 2
        result = create_diff_mask(img1, img2, open_iter=0, squaring=True, max_square=1)
        correct = np.zeros((n, n), dtype=bool)
        correct[20:69, 33:85] = True

        self.assertTrue(np.array_equal(result, correct))

        result = create_diff_mask(img1, img2, open_iter=0, squaring=True, max_square=1 / 4)
        correct = np.array(img1 > 1)

        self.assertTrue(np.array_equal(result, correct))

        result = create_diff_mask(img1, img2, open_iter=3, squaring=True, max_square=1)
        correct = np.zeros((n, n), dtype=bool)
        correct[39:59, 39:59] = True

        self.assertTrue(np.array_equal(result, correct))

    def test_find_xy_pos(self):
        n = 100
        img1 = np.ones((n, n))
        img2 = np.ones((n, n))

        img1[39, 59] = 2

        result = find_x_or_y_pos_milling_site(img1, img2, ax='x')
        correct = 59
        self.assertEqual(result, correct)

        result = find_x_or_y_pos_milling_site(img1, img2, ax='y')
        correct = 39
        self.assertEqual(result, correct)

        img1[30:51, 50:71] = 5
        img1 = gaussian_filter(img1, 5)

        result = find_x_or_y_pos_milling_site(img1, img2, ax='x')
        correct = 60
        self.assertEqual(result, correct)

        result = find_x_or_y_pos_milling_site(img1, img2, ax='y')
        correct = 40
        self.assertEqual(result, correct)

        img1[:, :5] = 10
        img1[:5, :] = 10
        img1[:, -5:] = 10
        img1[-5:, :] = 10

        result = find_x_or_y_pos_milling_site(img1, img2, ax='x')
        correct = 60
        self.assertEqual(result, correct)

        result = find_x_or_y_pos_milling_site(img1, img2, ax='y')
        correct = 40
        self.assertEqual(result, correct)

    def test_find_lines(self):
        # image = np.zeros((n, n))
        # for points in lines2:
        #     # Extracted points nested in the list
        #     x1, y1, x2, y2 = points[0]
        #     # Draw the lines joining the points on the original image
        #     cv2.line(image, (x1, y1), (x2, y2), 255, 1)

        n = 1000
        img = np.zeros((n, n))
        img[100:200, 200:400] = 1
        img[400:550, 300:400] = 1
        lines2 = np.zeros((5, 1, 4), dtype=int)

        # dit aanpassen naar middelpunten en angles en dan testen met een tolerantie

        lines2[0, 0, :] = np.array([200, 200, 200, 100])
        lines2[1, 0, :] = np.array([300, 550, 300, 400])
        lines2[2, 0, :] = np.array([400, 200, 400, 100])
        lines2[3, 0, :] = np.array([200, 100, 400, 100])
        lines2[4, 0, :] = np.array([200, 200, 400, 200])

        x_lines2 = (lines2.T[2] / 2 + lines2.T[0] / 2).T.reshape((len(lines2),))
        y_lines2 = (lines2.T[3] / 2 + lines2.T[1] / 2).T.reshape((len(lines2),))
        angle_lines2 = calculate_angles(lines2)

        x_lines, y_lines, lines, edges = find_lines(img, blur=10, low_thres_edges=0.1, high_thres_edges=2.5,
                                                    angle_res=np.pi / 180, line_thres=1, min_len_lines=1 / 15,
                                                    max_line_gap=1 / 50)
        angle_lines = calculate_angles(lines)

        self.assertTrue(np.sum(np.abs(x_lines - x_lines2)) < n / 20)
        self.assertTrue(np.sum(np.abs(y_lines - y_lines2)) < n / 20)
        self.assertTrue(np.sum(np.abs(angle_lines - angle_lines2)) < np.pi / 10 * len(lines))

        x_lines, y_lines, lines, edges = find_lines(img, blur=10, low_thres_edges=0.1, high_thres_edges=2.5,
                                                    angle_res=np.pi / 180, line_thres=1, min_len_lines=1 / 2,
                                                    max_line_gap=1 / 50)

        self.assertEqual(lines, None)

        img = np.zeros((n, n))
        img[100:200, 200:400] = 1
        img[100:200, 450:650] = 1

        x_lines, y_lines, lines, edges = find_lines(img, blur=10, low_thres_edges=0.1, high_thres_edges=2.5,
                                                    angle_res=np.pi / 180, line_thres=1, min_len_lines=1 / 4,
                                                    max_line_gap=1 / 10)
        angle_lines = calculate_angles(lines)

        lines2 = np.zeros((2, 1, 4), dtype=int)
        lines2[0, 0, :] = np.array([200, 100, 650, 100])
        lines2[1, 0, :] = np.array([200, 200, 650, 200])
        x_lines2 = (lines2.T[2] / 2 + lines2.T[0] / 2).T.reshape((len(lines2),))
        y_lines2 = (lines2.T[3] / 2 + lines2.T[1] / 2).T.reshape((len(lines2),))
        angle_lines2 = calculate_angles(lines2)

        self.assertTrue(np.sum(np.abs(x_lines - x_lines2)) < n / 20)
        self.assertTrue(np.sum(np.abs(y_lines - y_lines2)) < n / 20)
        self.assertTrue(np.sum(np.abs(angle_lines - angle_lines2)) < np.pi / 10 * len(lines))

        x_lines, y_lines, lines, edges = find_lines(img, blur=10, low_thres_edges=0.1, high_thres_edges=2.5,
                                                    angle_res=np.pi / 180, line_thres=1, min_len_lines=1 / 4,
                                                    max_line_gap=1 / 50)

        self.assertEqual(lines, None)

    def test_calculate_angles(self):
        lines = np.zeros((7, 1, 4))
        lines[0, 0, :] = np.array([0, 0, 1, 1])
        lines[1, 0, :] = np.array([0, 0, 0, 1])
        lines[2, 0, :] = np.array([0, 0, -1, 1])
        lines[3, 0, :] = np.array([0, 0, -1, -1])
        lines[4, 0, :] = np.array([0, 0, 1, -1])
        lines[5, 0, :] = np.array([0, 0, 1, 0])
        lines[6, 0, :] = np.array([23, 399, 433, 93])

        result = calculate_angles(lines)
        a = np.arctan((93 - 399) / (433 - 23)) + np.pi
        correct = np.array([np.pi / 4, np.pi / 2, np.pi * 3 / 4, np.pi / 4, np.pi * 3 / 4, 0, a])

        self.assertTrue(np.array_equal(result, correct))

    def test_combine_and_constraint_lines(self):
        # image = np.zeros((n, n))
        # for points in lines2:
        #     # Extracted points nested in the list
        #     x1, y1, x2, y2 = points[0]
        #     # Draw the lines joining the points on the original image
        #     cv2.line(image, (x1, y1), (x2, y2), 255, 1)
        x_lines2 = np.array([15, 35, 55, 25, 45, 65])
        y_lines2 = np.array([15, 35, 55, 15, 35, 55])
        lines2 = np.zeros((6, 1, 4), dtype=int)
        lines2[0, 0, :] = np.array([10, 10, 20, 20])
        lines2[1, 0, :] = np.array([30, 30, 40, 40])
        lines2[2, 0, :] = np.array([50, 50, 60, 60])
        lines2[3, 0, :] = np.array([20, 10, 30, 20])
        lines2[4, 0, :] = np.array([40, 30, 50, 40])
        lines2[5, 0, :] = np.array([60, 50, 70, 60])
        angle_lines2 = np.zeros(6) + np.pi / 4
        mid_milling_site = 50
        img = np.zeros((100, 100))
        img_shape = img.shape
        x_lines, y_lines, lines, angle_lines = combine_and_constraint_lines(x_lines2, y_lines2, lines2, angle_lines2,
                                                                            mid_milling_site, img_shape,
                                                                            x_width_constraint=1, y_width_constraint=1,
                                                                            angle_constraint=np.pi / 100,
                                                                            max_dist=1 / 9, max_diff_angle=np.pi / 10)

        # x_lines_correct = np.array([20, 40, 60])
        # y_lines_correct = np.array([15, 35, 55])
        lines_correct = np.zeros((3, 1, 4), dtype=int)
        lines_correct[0, 0, :] = np.array([15, 10, 25, 20])
        lines_correct[1, 0, :] = np.array([35, 30, 45, 40])
        lines_correct[2, 0, :] = np.array([55, 50, 65, 60])

        self.assertTrue(np.array_equal(lines, lines_correct))

        x_lines, y_lines, lines, angle_lines = combine_and_constraint_lines(x_lines2, y_lines2, lines2, angle_lines2,
                                                                            mid_milling_site, img_shape,
                                                                            x_width_constraint=1, y_width_constraint=1,
                                                                            angle_constraint=np.pi / 100,
                                                                            max_dist=1 / 11, max_diff_angle=np.pi / 10)

        lines_correct = np.zeros((6, 1, 4), dtype=int)
        lines_correct[0, 0, :] = np.array([10, 10, 20, 20])
        lines_correct[1, 0, :] = np.array([30, 30, 40, 40])
        lines_correct[2, 0, :] = np.array([50, 50, 60, 60])
        lines_correct[3, 0, :] = np.array([20, 10, 30, 20])
        lines_correct[4, 0, :] = np.array([40, 30, 50, 40])
        lines_correct[5, 0, :] = np.array([60, 50, 70, 60])

        self.assertTrue(np.array_equal(lines, lines_correct))

        x_lines, y_lines, lines, angle_lines = combine_and_constraint_lines(x_lines2, y_lines2, lines2, angle_lines2,
                                                                            mid_milling_site, img_shape,
                                                                            x_width_constraint=1,
                                                                            y_width_constraint=1 / 2,
                                                                            angle_constraint=np.pi / 100,
                                                                            max_dist=1 / 9, max_diff_angle=np.pi / 10)

        lines_correct = np.zeros((2, 1, 4), dtype=int)
        lines_correct[0, 0, :] = np.array([35, 30, 45, 40])
        lines_correct[1, 0, :] = np.array([55, 50, 65, 60])

        self.assertTrue(np.array_equal(lines, lines_correct))

        x_lines, y_lines, lines, angle_lines = combine_and_constraint_lines(x_lines2, y_lines2, lines2, angle_lines2,
                                                                            mid_milling_site, img_shape,
                                                                            x_width_constraint=1 / 3,
                                                                            y_width_constraint=1,
                                                                            angle_constraint=np.pi / 100,
                                                                            max_dist=1 / 9, max_diff_angle=np.pi / 10)

        lines_correct = np.zeros((2, 1, 4), dtype=int)
        lines_correct[0, 0, :] = np.array([35, 30, 45, 40])
        lines_correct[1, 0, :] = np.array([55, 50, 65, 60])

        self.assertTrue(np.array_equal(lines, lines_correct))

        x_lines2 = np.array([15, 10, 50, 51, 30, 32.5])
        y_lines2 = np.array([15, 15, 55, 55, 35, 35])
        lines2 = np.zeros((6, 1, 4), dtype=int)
        lines2[0, 0, :] = np.array([10, 10, 20, 20])
        lines2[1, 0, :] = np.array([10, 10, 10, 20])
        lines2[2, 0, :] = np.array([50, 50, 50, 60])
        lines2[3, 0, :] = np.array([50, 50, 52, 60])
        lines2[4, 0, :] = np.array([30, 30, 30, 40])
        lines2[5, 0, :] = np.array([30, 30, 35, 40])
        angle_lines2 = calculate_angles(lines2)
        mid_milling_site = 50
        img = np.zeros((100, 100))
        img_shape = img.shape

        x_lines, y_lines, lines, angle_lines = combine_and_constraint_lines(x_lines2, y_lines2, lines2, angle_lines2,
                                                                            mid_milling_site, img_shape,
                                                                            x_width_constraint=1, y_width_constraint=1,
                                                                            angle_constraint=np.pi / 100,
                                                                            max_dist=1 / 12, max_diff_angle=np.pi / 3)

        lines_correct = np.zeros((3, 1, 4), dtype=int)
        lines_correct[0, 0, :] = np.array([10, 10, 15, 20])
        lines_correct[1, 0, :] = np.array([50, 50, 51, 60])
        lines_correct[2, 0, :] = np.array([30, 30, 32.5, 40])
        self.assertTrue(np.array_equal(lines, lines_correct))

        x_lines, y_lines, lines, angle_lines = combine_and_constraint_lines(x_lines2, y_lines2, lines2, angle_lines2,
                                                                            mid_milling_site, img_shape,
                                                                            x_width_constraint=1, y_width_constraint=1,
                                                                            angle_constraint=np.pi / 100,
                                                                            max_dist=1 / 12, max_diff_angle=np.pi / 5)

        lines_correct = np.zeros((4, 1, 4), dtype=int)
        lines_correct[0, 0, :] = np.array([10, 10, 20, 20])
        lines_correct[1, 0, :] = np.array([10, 10, 10, 20])
        lines_correct[2, 0, :] = np.array([50, 50, 51, 60])
        lines_correct[3, 0, :] = np.array([30, 30, 32.5, 40])
        self.assertTrue(np.array_equal(lines, lines_correct))

        x_lines, y_lines, lines, angle_lines = combine_and_constraint_lines(x_lines2, y_lines2, lines2, angle_lines2,
                                                                            mid_milling_site, img_shape,
                                                                            x_width_constraint=1, y_width_constraint=1,
                                                                            angle_constraint=np.pi / 100,
                                                                            max_dist=1 / 12, max_diff_angle=np.pi / 9)

        lines_correct = np.zeros((5, 1, 4), dtype=int)
        lines_correct[0, 0, :] = np.array([10, 10, 20, 20])
        lines_correct[1, 0, :] = np.array([10, 10, 10, 20])
        lines_correct[2, 0, :] = np.array([50, 50, 51, 60])
        lines_correct[3, 0, :] = np.array([30, 30, 30, 40])
        lines_correct[4, 0, :] = np.array([30, 30, 35, 40])
        self.assertTrue(np.array_equal(lines, lines_correct))

        x_lines, y_lines, lines, angle_lines = combine_and_constraint_lines(x_lines2, y_lines2, lines2, angle_lines2,
                                                                            mid_milling_site, img_shape,
                                                                            x_width_constraint=1, y_width_constraint=1,
                                                                            angle_constraint=np.pi / 100,
                                                                            max_dist=1 / 12, max_diff_angle=np.pi / 17)

        lines_correct = np.zeros((6, 1, 4), dtype=int)
        lines_correct[0, 0, :] = np.array([10, 10, 20, 20])
        lines_correct[1, 0, :] = np.array([10, 10, 10, 20])
        lines_correct[2, 0, :] = np.array([50, 50, 50, 60])
        lines_correct[3, 0, :] = np.array([50, 50, 52, 60])
        lines_correct[4, 0, :] = np.array([30, 30, 30, 40])
        lines_correct[5, 0, :] = np.array([30, 30, 35, 40])
        self.assertTrue(np.array_equal(lines, lines_correct))

    def test_group_single_lines(self):
        # image = np.zeros((n, n))
        # for points in lines2:
        #     # Extracted points nested in the list
        #     x1, y1, x2, y2 = points[0]
        #     # Draw the lines joining the points on the original image
        #     cv2.line(image, (x1, y1), (x2, y2), 255, 1)
        x_lines2 = np.array([15, 35, 55, 25, 45, 65])
        y_lines2 = np.array([15, 35, 55, 15, 35, 55])
        lines2 = np.zeros((6, 1, 4), dtype=int)
        lines2[0, 0, :] = np.array([10, 10, 20, 20])
        lines2[1, 0, :] = np.array([30, 30, 40, 40])
        lines2[2, 0, :] = np.array([50, 50, 60, 60])
        lines2[3, 0, :] = np.array([20, 10, 30, 20])
        lines2[4, 0, :] = np.array([40, 30, 50, 40])
        lines2[5, 0, :] = np.array([60, 50, 70, 60])
        angle_lines2 = np.zeros(6) + np.pi / 4

        groups = group_single_lines(x_lines2, y_lines2, lines2, angle_lines2, max_distance=5)

        groups_correct = [[0, 1, 2], [3, 4, 5]]

        self.assertTrue(np.array_equal(groups, groups_correct))

        x_lines2 = np.array([12.5, 10, 35, 25, 20, 65])
        y_lines2 = np.array([15, 15, 35, 15, 15, 65])
        lines2 = np.zeros((6, 1, 4), dtype=int)
        lines2[0, 0, :] = np.array([10, 10, 15, 20])
        lines2[1, 0, :] = np.array([10, 10, 10, 20])
        lines2[2, 0, :] = np.array([30, 30, 40, 40])
        lines2[3, 0, :] = np.array([20, 10, 30, 20])
        lines2[4, 0, :] = np.array([20, 10, 20, 20])
        lines2[5, 0, :] = np.array([60, 50, 70, 80])
        angle_lines2 = calculate_angles(lines2)

        groups = group_single_lines(x_lines2, y_lines2, lines2, angle_lines2, max_distance=8, max_angle_diff=np.pi / 4)

        groups_correct = [[0, 1], [2, 3, 4], [5]]

        self.assertTrue(np.array_equal(groups, groups_correct))

        groups = group_single_lines(x_lines2, y_lines2, lines2, angle_lines2, max_distance=5, max_angle_diff=np.pi / 4)

        groups_correct = [[0, 1], [2, 4], [3, 4], [5]]

        self.assertTrue(np.array_equal(groups, groups_correct))

        groups = group_single_lines(x_lines2, y_lines2, lines2, angle_lines2, max_distance=5)

        groups_correct = [[0], [1], [2], [3], [4], [5]]

        self.assertTrue(np.array_equal(groups, groups_correct))

        groups = group_single_lines(x_lines2, y_lines2, lines2, angle_lines2, max_distance=8)

        groups_correct = [[0], [1], [2, 3], [4], [5]]

        self.assertTrue(np.array_equal(groups, groups_correct))

    def test_combine_groups(self):
        # image = np.zeros((n, n))
        # for points in lines2:
        #     # Extracted points nested in the list
        #     x1, y1, x2, y2 = points[0]
        #     # Draw the lines joining the points on the original image
        #     cv2.line(image, (x1, y1), (x2, y2), 255, 1)
        x_lines2 = np.array([15, 35, 55, 25, 45, 65, 35, 55, 75])
        y_lines2 = np.array([15, 35, 55, 15, 35, 55, 15, 35, 55])
        lines2 = np.zeros((9, 1, 4), dtype=int)
        lines2[0, 0, :] = np.array([10, 10, 20, 20])
        lines2[1, 0, :] = np.array([30, 30, 40, 40])
        lines2[2, 0, :] = np.array([50, 50, 60, 60])
        lines2[3, 0, :] = np.array([20, 10, 30, 20])
        lines2[4, 0, :] = np.array([40, 30, 50, 40])
        lines2[5, 0, :] = np.array([60, 50, 70, 60])
        lines2[6, 0, :] = np.array([30, 10, 40, 20])
        lines2[7, 0, :] = np.array([50, 30, 60, 40])
        lines2[8, 0, :] = np.array([70, 50, 80, 60])
        angle_lines2 = np.zeros(9) + np.pi / 4

        groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        size_groups_combined = combine_groups(groups, x_lines2, y_lines2, angle_lines2, min_dist=5, max_dist=10,
                                              max_angle_diff=np.pi / 8)
        size_groups_combined_correct = np.zeros((3, 3))
        size_groups_combined_correct[0, 1] = 6
        size_groups_combined_correct[1, 2] = 6

        self.assertTrue(np.array_equal(size_groups_combined, size_groups_combined_correct))

        lines2[6, 0, :] = np.array([35, 10, 45, 20])
        lines2[7, 0, :] = np.array([55, 30, 65, 40])
        lines2[8, 0, :] = np.array([75, 50, 85, 60])
        x_lines2 = np.array([15, 35, 55, 25, 45, 65, 40, 60, 80])
        y_lines2 = np.array([15, 35, 55, 15, 35, 55, 15, 35, 55])

        size_groups_combined = combine_groups(groups, x_lines2, y_lines2, angle_lines2, min_dist=5, max_dist=12,
                                              max_angle_diff=np.pi / 8)
        size_groups_combined_correct = np.zeros((3, 3))
        size_groups_combined_correct[0, 1] = 6
        size_groups_combined_correct[1, 2] = 6

        self.assertTrue(np.array_equal(size_groups_combined, size_groups_combined_correct))

        size_groups_combined = combine_groups(groups, x_lines2, y_lines2, angle_lines2, min_dist=5, max_dist=10,
                                              max_angle_diff=np.pi / 8)
        size_groups_combined_correct = np.zeros((3, 3))
        size_groups_combined_correct[0, 1] = 6

        self.assertTrue(np.array_equal(size_groups_combined, size_groups_combined_correct))

        size_groups_combined = combine_groups(groups, x_lines2, y_lines2, angle_lines2, min_dist=8, max_dist=12,
                                              max_angle_diff=np.pi / 8)
        size_groups_combined_correct = np.zeros((3, 3))
        size_groups_combined_correct[1, 2] = 6

        self.assertTrue(np.array_equal(size_groups_combined, size_groups_combined_correct))

    def test_combine_biggest_groups(self):
        # image = np.zeros((n, n))
        # for points in lines2:
        #     # Extracted points nested in the list
        #     x1, y1, x2, y2 = points[0]
        #     # Draw the lines joining the points on the original image
        #     cv2.line(image, (x1, y1), (x2, y2), 255, 1)
        x_lines2 = np.array([15, 35, 55, 25, 45, 65, 35, 55, 75])
        y_lines2 = np.array([15, 35, 55, 15, 35, 55, 15, 35, 55])
        lines2 = np.zeros((9, 1, 4), dtype=int)
        lines2[0, 0, :] = np.array([10, 10, 20, 20])
        lines2[1, 0, :] = np.array([30, 30, 40, 40])
        lines2[2, 0, :] = np.array([50, 50, 60, 60])
        lines2[3, 0, :] = np.array([20, 10, 30, 20])
        lines2[4, 0, :] = np.array([40, 30, 50, 40])
        lines2[5, 0, :] = np.array([60, 50, 70, 60])
        lines2[6, 0, :] = np.array([30, 10, 40, 20])
        lines2[7, 0, :] = np.array([50, 30, 60, 40])
        lines2[8, 0, :] = np.array([70, 50, 80, 60])
        angle_lines2 = np.zeros(9) + np.pi / 4

        groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        size_groups_combined = np.zeros((3, 3))
        size_groups_combined[0, 1] = 6
        size_groups_combined[1, 2] = 6
        biggest = np.where(size_groups_combined == np.max(size_groups_combined))

        merge_big_groups = combine_biggest_groups(biggest, groups, x_lines2, y_lines2, angle_lines2, max_dist=5,
                                                  max_angle_diff=np.pi/8)

        merge_big_groups_correct = np.zeros((2, 2))

        self.assertTrue(np.array_equal(merge_big_groups, merge_big_groups_correct))

        merge_big_groups = combine_biggest_groups(biggest, groups, x_lines2, y_lines2, angle_lines2, max_dist=10,
                                                  max_angle_diff=np.pi/8)

        merge_big_groups_correct = np.zeros((2, 2))
        merge_big_groups_correct[0, 1] = 12

        self.assertTrue(np.array_equal(merge_big_groups, merge_big_groups_correct))

        x_lines2 = np.array([10, 10, 10, 22.5, 32.5, 42.5, 35, 55, 75])
        y_lines2 = np.array([15, 35, 55, 15, 35, 55, 15, 35, 55])
        lines2 = np.zeros((9, 1, 4), dtype=int)
        lines2[0, 0, :] = np.array([10, 10, 10, 20])
        lines2[1, 0, :] = np.array([10, 30, 10, 40])
        lines2[2, 0, :] = np.array([10, 50, 10, 60])
        lines2[3, 0, :] = np.array([20, 10, 25, 20])
        lines2[4, 0, :] = np.array([30, 30, 35, 40])
        lines2[5, 0, :] = np.array([40, 50, 45, 60])
        lines2[6, 0, :] = np.array([30, 10, 40, 20])
        lines2[7, 0, :] = np.array([50, 30, 60, 40])
        lines2[8, 0, :] = np.array([70, 50, 80, 60])
        angle_lines2 = calculate_angles(lines2)

        groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        size_groups_combined = np.zeros((3, 3))
        size_groups_combined[0, 1] = 6
        size_groups_combined[1, 2] = 6
        biggest = np.where(size_groups_combined == np.max(size_groups_combined))

        merge_big_groups = combine_biggest_groups(biggest, groups, x_lines2, y_lines2, angle_lines2, max_dist=25,
                                                  max_angle_diff=np.pi/8)

        merge_big_groups_correct = np.zeros((2, 2))
        merge_big_groups_correct[0, 1] = 12

        self.assertTrue(np.array_equal(merge_big_groups, merge_big_groups_correct))

        merge_big_groups = combine_biggest_groups(biggest, groups, x_lines2, y_lines2, angle_lines2, max_dist=25,
                                                  max_angle_diff=np.pi / 10)

        merge_big_groups_correct = np.zeros((2, 2))

        self.assertTrue(np.array_equal(merge_big_groups, merge_big_groups_correct))

    def test_last_group_selection(self):
        # image = np.zeros((n, n))
        # for points in lines2:
        #     # Extracted points nested in the list
        #     x1, y1, x2, y2 = points[0]
        #     # Draw the lines joining the points on the original image
        #     cv2.line(image, (x1, y1), (x2, y2), 255, 1)
        x_lines2 = np.array([15, 35, 55, 25, 45, 65, 35, 55, 75])
        y_lines2 = np.array([15, 35, 55, 15, 35, 55, 15, 35, 55])
        lines2 = np.zeros((9, 1, 4), dtype=int)
        lines2[0, 0, :] = np.array([10, 10, 20, 20])
        lines2[1, 0, :] = np.array([30, 30, 40, 40])
        lines2[2, 0, :] = np.array([50, 50, 60, 60])
        lines2[3, 0, :] = np.array([20, 10, 30, 20])
        lines2[4, 0, :] = np.array([40, 30, 50, 40])
        lines2[5, 0, :] = np.array([60, 50, 70, 60])
        lines2[6, 0, :] = np.array([30, 10, 40, 20])
        lines2[7, 0, :] = np.array([50, 30, 60, 40])
        lines2[8, 0, :] = np.array([70, 50, 80, 60])
        angle_lines2 = np.zeros(9) + np.pi / 4

        groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        size_groups_combined = np.zeros((3, 3))
        size_groups_combined[0, 1] = 6
        size_groups_combined[1, 2] = 6
        biggest = np.where(size_groups_combined == np.max(size_groups_combined))
        merge_big_groups = np.zeros((2, 2))
        merge_big_groups[0, 1] = 12

        after_grouping = last_group_selection(groups, biggest, merge_big_groups, x_lines2, x_pos_mil=40,
                                              angle_lines=angle_lines2)

        after_grouping_correct = np.arange(9)

        self.assertTrue(np.array_equal(after_grouping, after_grouping_correct))

        merge_big_groups = np.zeros((2, 2))

        after_grouping = last_group_selection(groups, biggest, merge_big_groups, x_lines2, x_pos_mil=40,
                                              angle_lines=angle_lines2)

        after_grouping_correct = np.arange(6)

        self.assertTrue(np.array_equal(after_grouping, after_grouping_correct))

        after_grouping = last_group_selection(groups, biggest, merge_big_groups, x_lines2, x_pos_mil=70,
                                              angle_lines=angle_lines2)

        after_grouping_correct = np.arange(6)+3

        self.assertTrue(np.array_equal(after_grouping, after_grouping_correct))

        x_lines2 = np.array([10, 10, 10, 20, 20, 20, 40, 40, 40, 50, 50, 50, 15, 15, 15, 45, 45, 45])
        y_lines2 = np.array([15, 35, 55, 15, 35, 55, 15, 35, 55, 15, 35, 55, 15, 35, 55, 15, 35, 55])
        lines2 = np.zeros((18, 1, 4), dtype=int)
        lines2[0, 0, :] = np.array([10, 10, 10, 20])
        lines2[1, 0, :] = np.array([10, 30, 10, 40])
        lines2[2, 0, :] = np.array([10, 50, 10, 60])
        lines2[3, 0, :] = np.array([20, 10, 20, 20])
        lines2[4, 0, :] = np.array([20, 30, 20, 40])
        lines2[5, 0, :] = np.array([20, 50, 20, 60])
        lines2[6, 0, :] = np.array([40, 10, 40, 20])
        lines2[7, 0, :] = np.array([40, 30, 40, 40])
        lines2[8, 0, :] = np.array([40, 50, 40, 60])
        lines2[9, 0, :] = np.array([50, 10, 50, 20])
        lines2[10, 0, :] = np.array([50, 30, 50, 40])
        lines2[11, 0, :] = np.array([50, 50, 50, 60])
        lines2[12, 0, :] = np.array([15, 10, 15, 20])
        lines2[13, 0, :] = np.array([15, 30, 15, 40])
        lines2[14, 0, :] = np.array([15, 50, 15, 60])
        lines2[15, 0, :] = np.array([45, 10, 45, 20])
        lines2[16, 0, :] = np.array([45, 30, 45, 40])
        lines2[17, 0, :] = np.array([45, 50, 45, 60])
        angle_lines2 = np.zeros(18) + np.pi / 2

        groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17]]
        size_groups_combined = np.zeros((6, 6))
        size_groups_combined[0, 4] = 6
        size_groups_combined[1, 4] = 6
        size_groups_combined[2, 5] = 6
        size_groups_combined[3, 5] = 6
        biggest = np.where(size_groups_combined == np.max(size_groups_combined))
        merge_big_groups = np.zeros((4, 4))
        merge_big_groups[0, 1] = 12
        merge_big_groups[2, 3] = 12

        after_grouping = last_group_selection(groups, biggest, merge_big_groups, x_lines2, x_pos_mil=70,
                                              angle_lines=angle_lines2)

        after_grouping_correct = np.arange(9)+6
        after_grouping_correct[-3:] += 3

        self.assertTrue(np.array_equal(after_grouping, after_grouping_correct))

        after_grouping = last_group_selection(groups, biggest, merge_big_groups, x_lines2, x_pos_mil=20,
                                              angle_lines=angle_lines2)

        after_grouping_correct = np.arange(9)
        after_grouping_correct[-3:] += 6

        self.assertTrue(np.array_equal(after_grouping, after_grouping_correct))



    def test_couple_groups_of_lines(self):
        x_lines2 = np.array([15, 35, 55, 25, 45, 65])
        y_lines2 = np.array([15, 35, 55, 15, 35, 55])
        lines2 = np.zeros((6, 1, 4), dtype=int)
        lines2[0, 0, :] = np.array([10, 10, 20, 20])
        lines2[1, 0, :] = np.array([30, 30, 40, 40])
        lines2[2, 0, :] = np.array([50, 50, 60, 60])
        lines2[3, 0, :] = np.array([20, 10, 30, 20])
        lines2[4, 0, :] = np.array([40, 30, 50, 40])
        lines2[5, 0, :] = np.array([60, 50, 70, 60])
        angle_lines2 = np.zeros(6) + np.pi / 4
        groups = [[0, 1, 2], [3, 4, 5]]

        after_grouping = couple_groups_of_lines(groups, x_lines2, y_lines2, angle_lines2, x_pos_mil=50, min_dist=5,
                                                max_dist=10, max_angle_diff=np.pi / 8)

        after_grouping_correct = np.arange(6)
        self.assertTrue(np.array_equal(after_grouping, after_grouping_correct))

        after_grouping = couple_groups_of_lines(groups, x_lines2, y_lines2, angle_lines2, x_pos_mil=50, min_dist=1,
                                                max_dist=5, max_angle_diff=np.pi / 8)

        after_grouping_correct = np.array([])
        self.assertTrue(np.array_equal(after_grouping, after_grouping_correct))

        groups = [[0, 4, 2], [3, 1, 5]]
        after_grouping = couple_groups_of_lines(groups, x_lines2, y_lines2, angle_lines2, x_pos_mil=50, min_dist=2,
                                                max_dist=10,
                                                max_angle_diff=np.pi / 8)

        after_grouping_correct = np.arange(6)
        self.assertTrue(np.array_equal(after_grouping, after_grouping_correct))

        after_grouping = couple_groups_of_lines(groups, x_lines2, y_lines2, angle_lines2, x_pos_mil=50, min_dist=5,
                                                max_dist=10,
                                                max_angle_diff=np.pi / 8)

        after_grouping_correct = np.array([])
        self.assertTrue(np.array_equal(after_grouping, after_grouping_correct))

        x_lines2 = np.array([10, 10, 10, 22.5, 32.5, 42.5])
        y_lines2 = np.array([15, 35, 55, 15, 35, 55])
        lines2 = np.zeros((6, 1, 4), dtype=int)
        lines2[0, 0, :] = np.array([10, 10, 10, 20])
        lines2[1, 0, :] = np.array([10, 30, 10, 40])
        lines2[2, 0, :] = np.array([10, 50, 10, 60])
        lines2[3, 0, :] = np.array([20, 10, 25, 20])
        lines2[4, 0, :] = np.array([30, 30, 35, 40])
        lines2[5, 0, :] = np.array([40, 50, 45, 60])
        angle_lines2 = np.zeros(6)
        angle_lines2[:3] = np.pi / 2
        angle_lines2[3:] = np.arctan(2)

        groups = [[0, 1, 2], [3, 4, 5]]

        after_grouping = couple_groups_of_lines(groups, x_lines2, y_lines2, angle_lines2, x_pos_mil=50, min_dist=15,
                                                max_dist=35,
                                                max_angle_diff=np.pi / 6)

        after_grouping_correct = np.arange(6)
        self.assertTrue(np.array_equal(after_grouping, after_grouping_correct))

        after_grouping = couple_groups_of_lines(groups, x_lines2, y_lines2, angle_lines2, x_pos_mil=50, min_dist=15,
                                                max_dist=35,
                                                max_angle_diff=np.pi / 9)

        after_grouping_correct = np.array([])
        self.assertTrue(np.array_equal(after_grouping, after_grouping_correct))

        x_lines2 = np.array([15, 35, 55, 25, 45, 65, 35, 55, 75])
        y_lines2 = np.array([15, 35, 55, 15, 35, 55, 15, 35, 55])
        lines2 = np.zeros((9, 1, 4), dtype=int)
        lines2[0, 0, :] = np.array([10, 10, 20, 20])
        lines2[1, 0, :] = np.array([30, 30, 40, 40])
        lines2[2, 0, :] = np.array([50, 50, 60, 60])
        lines2[3, 0, :] = np.array([20, 10, 30, 20])
        lines2[4, 0, :] = np.array([40, 30, 50, 40])
        lines2[5, 0, :] = np.array([60, 50, 70, 60])
        lines2[6, 0, :] = np.array([30, 10, 40, 20])
        lines2[7, 0, :] = np.array([50, 30, 60, 40])
        lines2[8, 0, :] = np.array([70, 50, 80, 60])
        angle_lines2 = np.zeros(9) + np.pi / 4

        groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        after_grouping = couple_groups_of_lines(groups, x_lines2, y_lines2, angle_lines2, x_pos_mil=50, min_dist=5,
                                                max_dist=10,
                                                max_angle_diff=np.pi / 8)
        after_grouping_correct = np.arange(6) + 3
        self.assertTrue(np.array_equal(after_grouping, after_grouping_correct))

        after_grouping = couple_groups_of_lines(groups, x_lines2, y_lines2, angle_lines2, x_pos_mil=25, min_dist=5,
                                                max_dist=10,
                                                max_angle_diff=np.pi / 8)
        after_grouping_correct = np.arange(6)
        self.assertTrue(np.array_equal(after_grouping, after_grouping_correct))

    def test_create_line_mask(self):
        n = 6
        after_grouping = np.arange(n, dtype=int)
        x_lines2 = np.array([15, 35, 55, 25, 45, 65])
        y_lines2 = np.array([15, 35, 55, 15, 35, 55])
        lines2 = np.zeros((n, 1, 4), dtype=int)
        lines2[0, 0, :] = np.array([10, 10, 20, 20])
        lines2[1, 0, :] = np.array([30, 30, 40, 40])
        lines2[2, 0, :] = np.array([50, 50, 60, 60])
        lines2[3, 0, :] = np.array([20, 10, 30, 20])
        lines2[4, 0, :] = np.array([40, 30, 50, 40])
        lines2[5, 0, :] = np.array([60, 50, 70, 60])
        angle_lines2 = np.zeros(n) + np.pi / 4

        img = np.zeros((100, 100))
        img_shape = img.shape

        result = create_line_mask(after_grouping, x_lines2, y_lines2, lines2, angle_lines2, img_shape)

        correct = np.zeros((100, 100), dtype=bool)
        for i in range(correct.shape[1]):
            correct[i, i:i + 11] = True

        overlap = np.sum(result & correct) / np.sum(result | correct)

        self.assertTrue(overlap >= 0.90)

        img = np.zeros((100, 100))
        img_shape = img.shape

        result = create_line_mask(np.array([]), x_lines2, y_lines2, lines2, angle_lines2, img_shape)
        correct = np.zeros((100, 100), dtype=bool)

        self.assertTrue(np.array_equal(result, correct))

        n = 6
        after_grouping = np.arange(n, dtype=int)
        x_lines2 = np.array([10, 10, 10, 22.5, 32.5, 42.5])
        y_lines2 = np.array([15, 35, 55, 15, 35, 55])
        lines2 = np.zeros((n, 1, 4), dtype=int)
        lines2[0, 0, :] = np.array([10, 10, 10, 20])
        lines2[1, 0, :] = np.array([10, 30, 10, 40])
        lines2[2, 0, :] = np.array([10, 50, 10, 60])
        lines2[3, 0, :] = np.array([20, 10, 25, 20])
        lines2[4, 0, :] = np.array([30, 30, 35, 40])
        lines2[5, 0, :] = np.array([40, 50, 45, 60])
        angle_lines2 = np.zeros(n)
        angle_lines2[:3] = np.pi / 2
        angle_lines2[3:] = np.arctan(2)

        img = np.zeros((100, 100))
        img_shape = img.shape

        result = create_line_mask(after_grouping, x_lines2, y_lines2, lines2, angle_lines2, img_shape)
        correct = np.zeros((100, 100), dtype=bool)
        e = 0
        for i in range(img_shape[1]):
            correct[i, 10:15 + int(e)] = True
            e += 1 / 2

        overlap = np.sum(result & correct) / np.sum(result | correct)

        self.assertTrue(overlap >= 0.90)

    def test_combine_masks(self):
        n = 100
        mask1 = np.zeros((n, n), dtype=bool)
        mask2 = np.zeros((n, n), dtype=bool)
        mask3 = np.zeros((n, n), dtype=bool)
        mask1[30:40, 9:18] = True
        mask2[:, 10:21] = True

        result, combined = combine_masks(mask1, mask2, mask3, thres_diff=70, thres_lines=5, y_cut_off=6, iter_dil=0,
                                         iter_er=0, squaring=True)
        correct = np.zeros((n, n), dtype=bool)
        correct[30:40, 10:21] = True

        self.assertTrue(np.array_equal(result, correct))

        result, combined = combine_masks(mask1, mask2, mask3, thres_diff=70, thres_lines=5, y_cut_off=6, iter_dil=5,
                                         iter_er=0, squaring=True)
        correct = np.zeros((n, n), dtype=bool)
        correct[25:45, 10:21] = True

        self.assertTrue(np.array_equal(result, correct))

        result, combined = combine_masks(mask1, mask2, mask3, thres_diff=70, thres_lines=5, y_cut_off=6, iter_dil=0,
                                         iter_er=2, squaring=True)
        correct = np.zeros((n, n), dtype=bool)
        correct[30:40, 12:19] = True

        self.assertTrue(np.array_equal(result, correct))

        result, combined = combine_masks(mask1, mask2, mask3, thres_diff=95, thres_lines=5, y_cut_off=6, iter_dil=0,
                                         iter_er=0, squaring=True)
        correct = np.zeros((n, n), dtype=bool)
        correct[30:40, 10:21] = True

        self.assertTrue(np.array_equal(result, correct))

        result, combined = combine_masks(mask1, mask2, mask3, thres_diff=70, thres_lines=55, y_cut_off=6, iter_dil=0,
                                         iter_er=0, squaring=True)
        correct = np.zeros((n, n), dtype=bool)
        correct[30:40, 10:21] = True

        self.assertTrue(np.array_equal(result, correct))

        result, combined = combine_masks(mask1, mask2, mask3, thres_diff=95, thres_lines=55, y_cut_off=6, iter_dil=0,
                                         iter_er=0, squaring=True)
        correct = np.zeros((n, n), dtype=bool)
        correct[30:40, 9:18] = True

        self.assertTrue(np.array_equal(result, correct))

        result, combined = combine_masks(mask1, mask2, mask3, thres_diff=70, thres_lines=5, y_cut_off=3, iter_dil=0,
                                         iter_er=0, squaring=True)
        correct = np.zeros((n, n), dtype=bool)
        correct[33:40, 10:21] = True

        self.assertTrue(np.array_equal(result, correct))

        result, combined = combine_masks(mask1, mask2, mask3, thres_diff=70, thres_lines=5, y_cut_off=2, iter_dil=0,
                                         iter_er=0, squaring=True)
        correct = np.zeros((n, n), dtype=bool)

        self.assertTrue(np.array_equal(result, correct))

        mask1 = np.zeros((n, n), dtype=bool)
        mask2 = np.zeros((n, n), dtype=bool)
        mask3 = np.zeros((n, n), dtype=bool)

        for i in range(20):
            mask1[20 + i, 20 + i:30 + i] = True
        mask2[:, 30:40] = True
        mask3[:, 20:40] = True

        result, combined = combine_masks(mask1, mask2, mask3, thres_diff=70, thres_lines=5, y_cut_off=6, iter_dil=0,
                                         iter_er=0, squaring=False)
        correct = np.zeros((n, n), dtype=bool)
        correct[20:40, 20:40] = True

        self.assertTrue(np.array_equal(result, correct))

    def test_create_masked_img(self):
        n = 100
        img = np.zeros((n, n))
        mask = np.zeros((n, n), dtype=bool)
        mask[20:50, 20:50] = True
        for i in range(img.shape[0]):
            img[i, :] = np.arange(n) + i

        result, extent = create_masked_img(img, mask, cropping=False)

        correct = np.zeros((n, n))
        ran = np.zeros(n, dtype=bool)
        ran[20:50] = True
        for i in range(30):
            correct[i + 20, :] = (np.arange(n) + i + 20) * ran
        correct[0, 0] = 1

        self.assertTrue(np.array_equal(correct, result))

        mask = np.zeros((n, n), dtype=bool)
        r = 20
        mid_y = 60
        mid_x = 50
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                mask[i, j] = (i - mid_y) ** 2 + (j - mid_x) ** 2 <= r ** 2

        result, extent = create_masked_img(img, mask, cropping=False)
        correct = img * mask
        correct[0, 0] = 1

        self.assertTrue(np.array_equal(result, correct))

        result, extent = create_masked_img(img, mask, cropping=True)
        correct = correct[40:81, 30:71]
        correct[0, 0] = 1

        self.assertTrue(np.array_equal(result, correct))

    def test_create_binary_end_image(self):
        n = 100
        img = np.zeros((n, n))
        mask = np.zeros((n, n), dtype=bool)
        img[15:26, 40:51] = 0.5
        mask[10:61, 10:61] = True

        result, result2 = create_binary_end_image(mask, img, threshold=0.25, open_close=True, rid_of_back_signal=True,
                                                  b1=1, b2=4, b3=6, b4=10)

        correct = img > 0.25

        self.assertTrue(np.array_equal(correct, result))
        self.assertTrue(np.array_equal(np.zeros((n, n), dtype=bool), result2))

        result, result2 = create_binary_end_image(mask, img, threshold=0.55, open_close=True, rid_of_back_signal=True,
                                                  b1=1, b2=4, b3=6, b4=10)
        self.assertTrue(np.array_equal(np.zeros((n, n), dtype=bool), result))
        self.assertTrue(np.array_equal(np.zeros((n, n), dtype=bool), result2))

        result, result2 = create_binary_end_image(mask, img, threshold=0.25, open_close=True, rid_of_back_signal=False,
                                                  b1=1, b2=4, b3=6, b4=10)
        self.assertTrue(np.array_equal(correct, result))
        self.assertTrue(np.array_equal(correct, result2))

        img = np.zeros((n, n))
        mask = np.zeros((n, n), dtype=bool)
        img[15:20, 40:45] = 0.5
        img[40:51, 20:31] = 0.5
        mask[10:61, 10:61] = True
        correct = np.zeros((n, n), dtype=bool)
        correct[40:51, 20:31] = True

        result, result2 = create_binary_end_image(mask, img, threshold=0.25, open_close=True, rid_of_back_signal=False,
                                                  b1=1, b2=4, b3=6, b4=10)
        self.assertTrue(np.array_equal(correct, result))
        self.assertTrue(np.array_equal(correct, result2))

        result, result2 = create_binary_end_image(mask, img, threshold=0.25, open_close=True, rid_of_back_signal=True,
                                                  b1=1, b2=2, b3=6, b4=10)
        self.assertTrue(np.array_equal(correct, result2))
        correct[15:20, 40:45] = True
        self.assertTrue(np.array_equal(correct, result))

    def test_detect_blobs(self):
        n = 100
        img = np.zeros((n, n))
        img[10:21, 20:31] = 1
        img[40:51, 40:61] = 1
        img[20:26, 70:91] = 1

        keypoints, yxr = detect_blobs(img, min_circ=0, max_circ=1.01, min_area=0, max_area=np.inf, min_in=0,
                                      max_in=1.01, min_con=0, max_con=1.01, plotting=False)
        yxr_correct = np.array([[45, 50, 9], [22, 80, 6], [15, 25, 5]])

        self.assertTrue(np.array_equal(yxr, yxr_correct))

        keypoints, yxr = detect_blobs(img, min_circ=0.85, max_circ=1, min_area=0, max_area=np.inf, min_in=0,
                                      max_in=1.01, min_con=0, max_con=1.01, plotting=False)

        yxr_correct = np.array([])
        self.assertTrue(np.array_equal(yxr, yxr_correct))

        keypoints, yxr = detect_blobs(img, min_circ=0, max_circ=1.01, min_area=110, max_area=np.inf, min_in=0,
                                      max_in=1.01, min_con=0, max_con=1.01, plotting=False)
        yxr_correct = np.array([[45, 50, 9]])

        self.assertTrue(np.array_equal(yxr, yxr_correct))

        keypoints, yxr = detect_blobs(img, min_circ=0, max_circ=1.01, min_area=0, max_area=np.inf, min_in=0.5,
                                      max_in=1.01, min_con=0, max_con=1.01, plotting=False)
        yxr_correct = np.array([[15, 25, 5]])

        self.assertTrue(np.array_equal(yxr, yxr_correct))

        keypoints, yxr = detect_blobs(img, min_circ=0, max_circ=1.01, min_area=0, max_area=np.inf, min_in=0.2,
                                      max_in=1.01, min_con=0, max_con=1.01, plotting=False)
        yxr_correct = np.array([[45, 50, 9], [15, 25, 5]])

        self.assertTrue(np.array_equal(yxr, yxr_correct))

        n = 100
        img = np.zeros((n, n))

        circle_x = 60
        circle_y = 75
        r = 10
        for i in range(n):
            for j in range(n):
                img[i, j] = (i - circle_y) ** 2 + (j - circle_x) ** 2 <= r ** 2
        for i in range(20):
            img[10:10 + i, 10 + i] = 1
        # img[10:21, 20:31] = 1
        img[40:51, 40:61] = 1
        img[40:46, 70:91] = 1

        keypoints, yxr = detect_blobs(img, min_circ=0, max_circ=1.01, min_area=0, max_area=np.inf, min_in=0,
                                      max_in=1.01, min_con=0, max_con=1.01, plotting=False)
        yxr_correct = np.array([[75, 60, 9], [42, 80, 6], [45, 50, 9], [16, 23, 7]])

        self.assertTrue(np.array_equal(yxr, yxr_correct))

        keypoints, yxr = detect_blobs(img, min_circ=0.6, max_circ=1.01, min_area=0, max_area=np.inf, min_in=0,
                                      max_in=1.01, min_con=0, max_con=1.01, plotting=False)
        yxr_correct = np.array([[75, 60, 9], [45, 50, 9]])

        self.assertTrue(np.array_equal(yxr, yxr_correct))

        keypoints, yxr = detect_blobs(img, min_circ=0.8, max_circ=1.01, min_area=0, max_area=np.inf, min_in=0,
                                      max_in=1.01, min_con=0, max_con=1.01, plotting=False)
        yxr_correct = np.array([[75, 60, 9]])

        self.assertTrue(np.array_equal(yxr, yxr_correct))


if __name__ == '__main__':
    unittest.main()
