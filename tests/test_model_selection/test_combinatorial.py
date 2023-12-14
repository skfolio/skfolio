import numpy as np
from skfolio.model_selection import CombinatorialPurgedCV


def assert_split_equal(split, res):
    for i, (train, tests) in enumerate(split):
        assert np.array_equal(train, res[i][0])
        for j, test in enumerate(tests):
            assert np.array_equal(test, res[i][1][j])


def test_combinatorial_purged_cv():
    X = np.random.randn(12, 2)

    cv = CombinatorialPurgedCV(n_folds=3, n_test_folds=2, purged_size=0, embargo_size=0)

    assert_split_equal(
        cv.split(X),
        [
            (
                np.array([8, 9, 10, 11]),
                [np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])],
            ),
            (
                np.array([4, 5, 6, 7]),
                [np.array([0, 1, 2, 3]), np.array([8, 9, 10, 11])],
            ),
            (
                np.array([0, 1, 2, 3]),
                [np.array([4, 5, 6, 7]), np.array([8, 9, 10, 11])],
            ),
        ],
    )
    assert cv.n_splits == 3
    assert cv.n_test_paths == 2
    assert np.array_equal(cv.test_set_index, np.array([[0, 1], [0, 2], [1, 2]]))
    assert np.array_equal(
        cv.binary_train_test_sets,
        np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
    )
    assert np.array_equal(cv.recombined_paths, np.array([[0, 1], [0, 2], [1, 2]]))
    assert np.array_equal(cv.get_path_ids(), np.array([[0, 0], [1, 0], [1, 1]]))

    assert cv.plot_train_test_folds()
    assert cv.plot_train_test_index(X)
    cv.summary(X)

    cv = CombinatorialPurgedCV(n_folds=3, n_test_folds=2, purged_size=1, embargo_size=0)
    assert_split_equal(
        cv.split(X),
        [
            (np.array([9, 10, 11]), [np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])]),
            (np.array([5, 6]), [np.array([0, 1, 2, 3]), np.array([8, 9, 10, 11])]),
            (np.array([0, 1, 2]), [np.array([4, 5, 6, 7]), np.array([8, 9, 10, 11])]),
        ],
    )

    cv = CombinatorialPurgedCV(n_folds=3, n_test_folds=2, purged_size=0, embargo_size=1)
    assert_split_equal(
        cv.split(X),
        [
            (np.array([9, 10, 11]), [np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])]),
            (np.array([5, 6, 7]), [np.array([0, 1, 2, 3]), np.array([8, 9, 10, 11])]),
            (
                np.array([0, 1, 2, 3]),
                [np.array([4, 5, 6, 7]), np.array([8, 9, 10, 11])],
            ),
        ],
    )

    cv = CombinatorialPurgedCV(n_folds=5, n_test_folds=2, purged_size=0, embargo_size=0)
    assert_split_equal(
        cv.split(X),
        [
            (
                np.array([4, 5, 6, 7, 8, 9, 10, 11]),
                [np.array([0, 1]), np.array([2, 3])],
            ),
            (
                np.array([2, 3, 6, 7, 8, 9, 10, 11]),
                [np.array([0, 1]), np.array([4, 5])],
            ),
            (
                np.array([2, 3, 4, 5, 8, 9, 10, 11]),
                [np.array([0, 1]), np.array([6, 7])],
            ),
            (
                np.array([2, 3, 4, 5, 6, 7]),
                [np.array([0, 1]), np.array([8, 9, 10, 11])],
            ),
            (
                np.array([0, 1, 6, 7, 8, 9, 10, 11]),
                [np.array([2, 3]), np.array([4, 5])],
            ),
            (
                np.array([0, 1, 4, 5, 8, 9, 10, 11]),
                [np.array([2, 3]), np.array([6, 7])],
            ),
            (
                np.array([0, 1, 4, 5, 6, 7]),
                [np.array([2, 3]), np.array([8, 9, 10, 11])],
            ),
            (
                np.array([0, 1, 2, 3, 8, 9, 10, 11]),
                [np.array([4, 5]), np.array([6, 7])],
            ),
            (
                np.array([0, 1, 2, 3, 6, 7]),
                [np.array([4, 5]), np.array([8, 9, 10, 11])],
            ),
            (
                np.array([0, 1, 2, 3, 4, 5]),
                [np.array([6, 7]), np.array([8, 9, 10, 11])],
            ),
        ],
    )
    assert cv.n_splits == 10
    assert cv.n_test_paths == 4
    assert np.array_equal(
        cv.test_set_index,
        np.array(
            [
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [1, 2],
                [1, 3],
                [1, 4],
                [2, 3],
                [2, 4],
                [3, 4],
            ]
        ),
    )
    assert np.array_equal(
        cv.binary_train_test_sets,
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            ]
        ),
    )
    assert np.array_equal(
        cv.recombined_paths,
        np.array(
            [[0, 1, 2, 3], [0, 4, 5, 6], [1, 4, 7, 8], [2, 5, 7, 9], [3, 6, 8, 9]]
        ),
    )
    assert np.array_equal(
        cv.get_path_ids(),
        np.array(
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [3, 0],
                [1, 1],
                [2, 1],
                [3, 1],
                [2, 2],
                [3, 2],
                [3, 3],
            ]
        ),
    )
