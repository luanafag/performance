import numpy as np
import pytest

from rbg.ModelSelection import KFold, StratifiedKFold, GroupKFold


class TestKFold:
    def test_invalid_k(self):
        X = np.arange(10).reshape(10, 1)
        for k_invalido in [0, 1, -1, 11, 2.5]:
            with pytest.raises(ValueError):
                cv = KFold(k=k_invalido)
                list(cv.split(X))

    def test_dataset_size(self):
        X = np.arange(10).reshape(10, 1)
        cv = KFold(k=3)
        folds = list(cv.split(X))
        assert len(folds) == 3
        all_test_indices = np.concatenate([test for train, test in folds])
        assert len(all_test_indices) == len(X)
        assert sorted(all_test_indices) == list(range(len(X)))

    def test_reproducibility(self):
        X = np.arange(20).reshape(20, 1)
        cv1 = KFold(k=4, shuffle=True, random_state=42)
        cv2 = KFold(k=4, shuffle=True, random_state=42)
        folds1 = list(cv1.split(X))
        folds2 = list(cv2.split(X))
        for (train1, test1), (train2, test2) in zip(folds1, folds2):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(test1, test2)

    def test_train_before_test(self):
        X = np.arange(6).reshape(6, 1)
        cv = KFold(k=3)
        train_idxs, test_idxs = next(cv.split(X))
        np.testing.assert_array_equal(test_idxs, [0, 1])
        np.testing.assert_array_equal(train_idxs, [2, 3, 4, 5])


class TestStratifiedKFold:

    def test_invalid_k(self):
        X = np.arange(10).reshape(10, 1)
        y = np.array(['A'] * 5 + ['B'] * 5)
        for k in [0, 1, -1, 11, 2.5]:
            with pytest.raises(ValueError):
                cv = StratifiedKFold(k=k)
                list(cv.split(X, y))

    def test_all_classes_in_folds(self):
        X = np.arange(30).reshape(30, 1)
        y = np.array(['A'] * 10 + ['B'] * 10 + ['C'] * 10)
        cv = StratifiedKFold(k=5)
        folds = list(cv.split(X, y))
        for train_idxs, test_idxs in folds:
            assert set(y[test_idxs]) == {'A', 'B', 'C'}
            assert set(y[train_idxs]) == {'A', 'B', 'C'}

    def test_stratification_balance(self):
        X = np.arange(30).reshape(30, 1)
        y = np.array(['A'] * 10 + ['B'] * 10 + ['C'] * 10)
        cv = StratifiedKFold(k=5)
        folds = list(cv.split(X, y))
        for label in np.unique(y):
            counts = [
                np.sum(y[test_idxs] == label)
                for train_idxs, test_idxs in folds
            ]

            assert max(counts) - min(counts) <= 1

    def test_reproducibility(self):
        X = np.arange(30).reshape(30, 1)
        y = np.array(['A'] * 10 + ['B'] * 10 + ['C'] * 10)
        cv1 = StratifiedKFold(k=5, shuffle=True, random_state=42)
        cv2 = StratifiedKFold(k=5, shuffle=True, random_state=42)
        folds1 = list(cv1.split(X, y))
        folds2 = list(cv2.split(X, y))
        for (train1, test1), (train2, test2) in zip(folds1, folds2):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(test1, test2)

class TestGroupKFold:
    def test_invalid_k(self):
        X = np.arange(10).reshape(10, 1)
        groups = np.array([
            'A', 'A',
            'B', 'B',
            'C', 'C',
            'D', 'D',
            'E', 'E'
        ])
        for k in [0, 1, -1, 6, 2.5]:
            with pytest.raises(ValueError):
                cv = GroupKFold(k=k)
                list(cv.split(X, groups))

    def test_number_of_groups(self):
        X = np.arange(10).reshape(10, 1)
        groups = np.array([
            'A', 'A',
            'B', 'B',
            'C', 'C',
            'D', 'D',
            'E', 'E'
        ])

        cv = GroupKFold(k=3)
        folds = list(cv.split(X, groups))
        assert len(folds) == 3
        all_test_groups = np.concatenate([
            np.unique(groups[test_idxs])
            for train_idxs, test_idxs in folds
        ])
        assert len(all_test_groups) == len(np.unique(groups))
        assert set(all_test_groups) == set(np.unique(groups))

    def test_groups_descending_order(self):
        group_sizes = {
            'A': 8,
            'B': 6,
            'C': 4,
            'D': 2
        }

        groups = np.concatenate([
            np.repeat(group, size)
            for group, size in group_sizes.items()
        ])
        X = np.arange(len(groups)).reshape(-1, 1)
        cv = GroupKFold(k=4)
        folds = list(cv.split(X, groups))
        assigned_sizes = []
        for train_idxs, test_idxs in folds:
            test_groups = np.unique(groups[test_idxs])
            assigned_sizes.append(group_sizes[test_groups[0]])
        assert assigned_sizes == [8, 6, 4, 2]

    def test_fold_sizes_balance(self):
        group_sizes = {
            'A': 8,
            'B': 7,
            'C': 6,
            'D': 5,
            'E': 4,
            'F': 3
        }

        groups = np.concatenate([
            np.repeat(group, size)
            for group, size in group_sizes.items()
        ])
        X = np.arange(len(groups)).reshape(-1, 1)
        cv = GroupKFold(k=3)
        folds = list(cv.split(X, groups))
        fold_sizes = [
            len(test_idxs)
            for train_idxs, test_idxs in folds
        ]
        assert max(fold_sizes) - min(fold_sizes) <= 2

    def test_no_group_leakage(self):
        X = np.arange(20).reshape(20, 1)
        groups = np.repeat(
            ['A', 'B', 'C', 'D', 'E'],
            4
        )
        cv = GroupKFold(k=3)
        folds = list(cv.split(X, groups))
        for train_idxs, test_idxs in folds:
            train_groups = set(groups[train_idxs])
            test_groups = set(groups[test_idxs])
            assert train_groups.isdisjoint(test_groups)

    def test_reproducibility(self):
        X = np.arange(30).reshape(30, 1)

        groups = np.repeat(
            ['A', 'B', 'C', 'D', 'E', 'F'],
            5
        )
        cv1 = GroupKFold(k=3, shuffle=True, random_state=42)
        cv2 = GroupKFold(k=3, shuffle=True, random_state=42)
        folds1 = list(cv1.split(X, groups))
        folds2 = list(cv2.split(X, groups))
        for (train1, test1), (train2, test2) in zip(folds1, folds2):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(test1, test2)