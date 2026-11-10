import unittest

import numpy as np

from rbg.Utilities.crossvalidation import train_test_split


class TrainTestSplitTests(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(22).reshape(11, 2)
        self.y = np.arange(11)

    def test_fractional_size_rounds_up_and_partitions_all_samples(self):
        for shuffle in (False, True):
            with self.subTest(shuffle=shuffle):
                X_train, X_test, y_train, y_test = train_test_split(
                    self.X, self.y, test_size=0.2, shuffle=shuffle, random_state=42
                )

                self.assertEqual(len(y_test), 3)
                self.assertEqual(len(y_train), 8)
                self.assertFalse(set(y_train) & set(y_test))
                np.testing.assert_array_equal(
                    np.sort(np.concatenate((y_train, y_test))), self.y
                )
                np.testing.assert_array_equal(X_train, self.X[y_train])
                np.testing.assert_array_equal(X_test, self.X[y_test])

    def test_integer_size_is_an_exact_count(self):
        for shuffle in (False, True):
            with self.subTest(shuffle=shuffle):
                _, _, y_train, y_test = train_test_split(
                    self.X, self.y, test_size=2, shuffle=shuffle, random_state=42
                )

                self.assertEqual(len(y_test), 2)
                self.assertEqual(len(y_train), 9)

    def test_shuffle_is_reproducible_with_a_seed(self):
        first = train_test_split(self.X, self.y, random_state=42)
        second = train_test_split(self.X, self.y, random_state=42)

        for actual, expected in zip(first, second):
            np.testing.assert_array_equal(actual, expected)

    def test_without_shuffle_preserves_order(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, shuffle=False
        )

        np.testing.assert_array_equal(X_test, self.X[:3])
        np.testing.assert_array_equal(X_train, self.X[3:])
        np.testing.assert_array_equal(y_test, self.y[:3])
        np.testing.assert_array_equal(y_train, self.y[3:])


if __name__ == '__main__':
    unittest.main()
