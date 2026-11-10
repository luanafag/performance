from rbg.ModelSelection import LOOCV
import numpy as np

class TestLOOCV:
    def test_test_size_is_one(self):
        X = np.arange(10).reshape(10, 1)
        cv = LOOCV()
        folds = list(cv.split(X))
        assert len(folds) == len(X)
        for train_idxs, test_idxs in folds:
            assert len(test_idxs) == 1
            assert len(train_idxs) == len(X) - 1

    def test_reproducibility(self):
        X = np.arange(10).reshape(10, 1)
        cv1 = LOOCV()
        cv2 = LOOCV()
        folds1 = list(cv1.split(X))
        folds2 = list(cv2.split(X))
        for (train1, test1), (train2, test2) in zip(folds1, folds2):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(test1, test2)