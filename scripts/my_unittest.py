import logging
import unittest
import gc
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from odemis.dataio import tiff
from scripts.functions_lars import find_focus_z_slice_and_outlier_cutoff, rescaling, overlay, \
    z_projection_and_outlier_cutoff, blur_and_norm, create_diff_mask, find_x_or_y_pos_milling_site, calculate_angles, \
    create_line_mask, combine_masks, create_masked_img, create_binary_end_image
from try_out_lars import data_paths_before, data_paths_after, data_paths_true_masks
from copy import deepcopy


class TestFunctionsLars(unittest.TestCase):
    """My test cases testing my tests"""

    @classmethod
    def setUpClass(cls):
        cls.path_before = data_paths_before
        cls.path_after = data_paths_after
        cls.path_masks = data_paths_true_masks

        ll = len(data_paths_before)
        cls.channel_before = np.zeros(ll, dtype=int)
        cls.channel_before[7] = 1
        cls.channel_after = np.zeros(ll, dtype=int)

        # found manually after scaling and overlay
        cls.correct_milling_pos_y = [558, 945, 1022, 1182, 1040, 1368, 1355, 1050, 430, 470, 490, 630, 1035, 640, 850,
                                     1114, 1025, 800]
        cls.correct_milling_pos_x = [918, 1291, 1673, 1458, 1535, 1355, 1355, 1144, 610, 930, 820, 630, 1140, 1800,
                                     1955, 950, 1035, 1850]

        cls.correct_z_slice_before = [19, 23, 23, 26, 33, 0, 23, 0, 27, 22, 22, 23, 26, 36, 23, 25, 24, 24]
        cls.correct_z_slice_after = [10, 15, 22, 19, 32, 0, 17, 2, 22, 18, 25, 22, 17, 15, 16, 17, 23, 24]

        cls.correct_shift = np.zeros((18, 2))

        # img_before size / img_after size
        cls.correct_scaling = np.ones(18)
        cls.correct_scaling[2] = 2
        cls.correct_scaling[17] = 0.5

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
                                                                    which_channel=self.channel_before[i],
                                                                    square_width=1 / 4,
                                                                    num_slices=5, outlier_cutoff=99, mode='max')

            else:
                img, result = find_focus_z_slice_and_outlier_cutoff(data,
                                                                    milling_pos_y=self.correct_milling_pos_y[i] *
                                                                                  self.correct_scaling[i],
                                                                    milling_pos_x=self.correct_milling_pos_x[i] *
                                                                                  self.correct_scaling[i],
                                                                    which_channel=self.channel_before[i],
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

            img = z_projection_and_outlier_cutoff(data, 30, which_channel=self.channel_before[i], outlier_cutoff=99,
                                                  mode='max')

            if data[0].shape[0] == 1:
                s = int((len(data[self.channel_before[i]][0][0]) - 30) / 2)
                img_correct = np.array(data[self.channel_before[i]][0][0][s:-s], dtype=int)
                histo, bins = np.histogram(img_correct, bins=2000)
                histo = np.cumsum(histo)
                histo = histo / histo[-1] * 100
                plc = np.min(np.where(histo > 99))
                thr_out = bins[plc] / 2 + bins[plc + 1] / 2
                img_correct[img_correct > thr_out] = thr_out

                img_correct = np.max(img_correct, axis=0)

            else:
                img_correct = np.array(data[self.channel_before[i]], dtype=int)
                histo, bins = np.histogram(img_correct, bins=2000)
                histo = np.cumsum(histo)
                histo = histo / histo[-1] * 100
                plc = np.min(np.where(histo > 99))
                thr_out = bins[plc] / 2 + bins[plc + 1] / 2
                img_correct[img_correct > thr_out] = thr_out

            self.assertTrue(np.array_equal(img, img_correct))
            # use np.testing.assert_almost_equal()

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

        correct_a = np.zeros((n*m, n*m))
        correct_a[0:m, :] = 1
        correct_b = np.ones((n*m, n*m))
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
        d = np.zeros(2*m)
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
        correct_a[:int(correct_a.shape[0]/2), :] = 1
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
            b[int(i*10), int(i*10)] = 1

            b_result, shift = overlay(a, b, 1)
            correct_shift = np.array([[50-int(i*10)], [50-int(i*10)]])

            self.assertTrue(np.array_equal(shift, correct_shift))

        for i in range(11):
            a = np.zeros((n, n))
            b = np.zeros((n, n))
            a[50, 50] = 1
            b[50, int(i*10)] = 1

            b_result, shift = overlay(a, b, 1)
            correct_shift = np.array([[0], [50-int(i*10)]])

            self.assertTrue(np.array_equal(shift, correct_shift))

        for i in range(11):
            a = np.zeros((n, n))
            b = np.zeros((n, n))
            a[50, 50] = 1
            b[int(i*10), 50] = 1

            b_result, shift = overlay(a, b, 1)
            correct_shift = np.array([[50-int(i*10)], [0]])

            self.assertTrue(np.array_equal(shift, correct_shift))

    def test_get_image(self):
        pass

    def test_blur_and_norm(self):
        img = np.zeros((100, 100))
        img[49, 49] = 1

        result, result_blurred = blur_and_norm(img, 1)

        correct = img
        correct_blurred = gaussian_filter(img, 1)
        correct_blurred = correct_blurred/np.max(correct_blurred)

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

        result = create_diff_mask(img1, img2, open_iter=0, squaring=True, max_square=1/4)
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
        pass

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
        a = np.arctan((93-399)/(433-23)) + np.pi
        correct = np.array([np.pi/4, np.pi/2, np.pi*3/4, np.pi/4, np.pi*3/4, 0, a])

        self.assertTrue(np.array_equal(result, correct))

    def test_combine_and_constraint_lines(self):
        pass

    def test_group_single_lines(self):
        pass

    def test_couple_groups_of_lines(self):
        pass

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
        angle_lines2 = np.zeros(n) + np.pi/4

        img = np.zeros((100, 100))
        img_shape = img.shape

        result = create_line_mask(after_grouping, x_lines2, y_lines2, lines2, angle_lines2, img_shape)

        correct = np.zeros((100, 100), dtype=bool)
        for i in range(correct.shape[1]):
            correct[i, i:i+11] = True

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
        angle_lines2[:3] = np.pi/2
        angle_lines2[3:] = np.arctan(2)

        img = np.zeros((100, 100))
        img_shape = img.shape

        result = create_line_mask(after_grouping, x_lines2, y_lines2, lines2, angle_lines2, img_shape)
        correct = np.zeros((100, 100), dtype=bool)
        e = 0
        for i in range(img_shape[1]):
            correct[i, 10:15+int(e)] = True
            e += 1/2

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
            mask1[20+i, 20+i:30+i] = True
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
            correct[i+20, :] = (np.arange(n) + i + 20) * ran
        correct[0, 0] = 1

        self.assertTrue(np.array_equal(correct, result))

        mask = np.zeros((n, n), dtype=bool)
        r = 20
        mid_y = 60
        mid_x = 50
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                mask[i, j] = (i-mid_y)**2+(j-mid_x)**2 <= r**2

        result, extent = create_masked_img(img, mask, cropping=False)
        correct = img*mask
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
        pass

if __name__ == '__main__':
    unittest.main()
