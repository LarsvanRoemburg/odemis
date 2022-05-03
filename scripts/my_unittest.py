import unittest
import numpy as np
from odemis.dataio import tiff
from scripts.functions_lars import find_focus_z_slice_and_outlier_cutoff, rescaling
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

    def test_rescaling(self):
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

    def test_focus_z_slice(self):
        for i in range(int(len(data_paths_after) / 3)):
            if i > 4:
                i = int(2 * i)
                print(i)

                # data = tiff.read_data(data_paths_after[i])
                # if self.correct_scaling[i] <= 1:
                #     result = find_focus_z_slice_and_outlier_cutoff(data, milling_pos_y=self.correct_milling_pos_y[i] -
                #                                                                           self.correct_shift[i][0],
                #                                                    milling_pos_x=self.correct_milling_pos_x[i] -
                #                                                                  self.correct_shift[i][1],
                #                                                    which_channel=self.channel_after[i], square_width=1 / 4,
                #                                                    num_slices=5, outlier_cutoff=99, mode='max')
                # else:
                #     result = find_focus_z_slice_and_outlier_cutoff(data, milling_pos_y=(self.correct_milling_pos_y[i] -
                #                                                                            self.correct_shift[i][0]) /
                #                                                                           self.correct_scaling[i],
                #                                                    milling_pos_x=(self.correct_milling_pos_x[i] -
                #                                                                   self.correct_shift[i][1]) /
                #                                                                  self.correct_scaling[i],
                #                                                    which_channel=self.channel_after[i], square_width=1 / 4,
                #                                                    num_slices=5, outlier_cutoff=99, mode='max')
                #
                # data = data[self.channel_after[i]][0][0]
                # histo, bins = np.histogram(data, bins=2000)
                # histo = np.cumsum(histo)
                # histo = histo / histo[-1] * 100
                # plc = np.min(np.where(histo > 99))
                # thr_out = bins[plc] / 2 + bins[plc + 1] / 2
                # data[data > thr_out] = thr_out
                # correct_img = np.array(np.max(data[int(self.correct_z_slice_after[i] - 2):int(self.correct_z_slice_after[i] + 3)], axis=0), dtype=int)
                # # np.array_equal
                # self.assertTrue(np.array_equal(correct_img, result))

                data = tiff.read_data(data_paths_before[i])
                if self.correct_scaling[i] >= 1:
                    result = find_focus_z_slice_and_outlier_cutoff(data, milling_pos_y=self.correct_milling_pos_y[i],
                                                                   milling_pos_x=self.correct_milling_pos_x[i],
                                                                   which_channel=self.channel_before[i],
                                                                   square_width=1 / 4,
                                                                   num_slices=5, outlier_cutoff=99, mode='max')
                else:
                    result = find_focus_z_slice_and_outlier_cutoff(data, milling_pos_y=self.correct_milling_pos_y[i] *
                                                                                       self.correct_scaling[i],
                                                                   milling_pos_x=self.correct_milling_pos_x[i] *
                                                                                 self.correct_scaling[i],
                                                                   which_channel=self.channel_before[i],
                                                                   square_width=1 / 4,
                                                                   num_slices=5, outlier_cutoff=99, mode='max')

                dat = deepcopy(data[self.channel_before[i]][0][0])
                histo, bins = np.histogram(dat, bins=2000)
                histo = np.cumsum(histo)
                histo = histo / histo[-1] * 100
                plc = np.min(np.where(histo > 99))
                thr_out = bins[plc] / 2 + bins[plc + 1] / 2
                dat[dat > thr_out] = thr_out
                correct_img = np.array(
                    np.max(dat[int(self.correct_z_slice_before[i] - 2):int(self.correct_z_slice_before[i] + 3)],
                           axis=0), dtype=int)
                # np.array_equal
                self.assertTrue(np.array_equal(correct_img, result))


if __name__ == '__main__':
    unittest.main()
