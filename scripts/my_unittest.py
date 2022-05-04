import unittest
import gc
import numpy as np
from odemis.dataio import tiff
from scripts.functions_lars import find_focus_z_slice_and_outlier_cutoff, rescaling, overlay, \
    z_projection_and_outlier_cutoff
from try_out_lars import data_paths_before, data_paths_after, data_paths_true_masks
from copy import deepcopy


class TestMyClass(unittest.TestCase):
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
        for i in range(int(len(data_paths_after) / 3)):
            i = 2*i
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
        for i in range(int(len(data_paths_after) / 3)):
            i = 2*i
            data = tiff.read_data(data_paths_before[i])

            img = z_projection_and_outlier_cutoff(data, 30, which_channel=self.channel_before[i], outlier_cutoff=99, mode='max')

            s = int((len(data[self.channel_before[i]][0][0]) - 30) / 2)
            img_correct = np.array(data[self.channel_before[i]][0][0][s:-s], dtype=int)

            histo, bins = np.histogram(img_correct, bins=2000)
            histo = np.cumsum(histo)
            histo = histo / histo[-1] * 100
            plc = np.min(np.where(histo > 99))
            thr_out = bins[plc] / 2 + bins[plc + 1] / 2
            img_correct[img_correct > thr_out] = thr_out

            img_correct = np.max(img_correct, axis=0)
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



if __name__ == '__main__':
    unittest.main()
