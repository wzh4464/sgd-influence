###
# File: /test_data_modules.py
# Created Date: Saturday, September 28th 2024
# Author: Zihan
# -----
# Last Modified: Saturday, 28th September 2024 12:51:02 am
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import unittest
from DataModule import fetch_data_module


class TestDataModules(unittest.TestCase):
    def setUp(self):
        self.sample_size = 256
        self.datasets = ["mnist", "20news", "adult", "cifar", "emnist"]

    def test_data_shapes(self):
        for dataset in self.datasets:
            with self.subTest(dataset=dataset):
                data_module = fetch_data_module(dataset)
                (x_tr, y_tr), (x_val, y_val), (x_test, y_test) = data_module.fetch(
                    self.sample_size, self.sample_size, self.sample_size, seed=42
                )

                self.assertEqual(len(x_tr), self.sample_size)
                self.assertEqual(len(y_tr), self.sample_size)

                if dataset == "mnist":
                    self.assertEqual(x_tr.shape, (256, 1, 28, 28))
                elif dataset == "20news":
                    self.assertEqual(x_tr.shape, (256, 4752))
                elif dataset == "adult":
                    self.assertEqual(x_tr.shape, (256, 62))
                elif dataset == "cifar":
                    self.assertEqual(x_tr.shape, (256, 32, 32, 3))
                elif dataset == "emnist":
                    self.assertEqual(x_tr.shape, (256, 1, 28, 28))

                self.assertEqual(y_tr.shape, (256,))

    def test_data_types(self):
        for dataset in self.datasets:
            with self.subTest(dataset=dataset):
                data_module = fetch_data_module(dataset)
                (x_tr, y_tr), (_, _), (_, _) = data_module.fetch(
                    self.sample_size, self.sample_size, self.sample_size, seed=42
                )

                self.assertTrue(x_tr.dtype.name.startswith("float"))
                self.assertTrue(
                    y_tr.dtype.name.startswith("int")
                    or y_tr.dtype.name.startswith("float")
                )

    def test_data_range(self):
        for dataset in self.datasets:
            with self.subTest(dataset=dataset):
                data_module = fetch_data_module(dataset)
                (x_tr, y_tr), (_, _), (_, _) = data_module.fetch(
                    self.sample_size, self.sample_size, self.sample_size, seed=42
                )

                self.assertTrue(set(y_tr) <= {0, 1})  # Binary classification


if __name__ == "__main__":
    unittest.main()
