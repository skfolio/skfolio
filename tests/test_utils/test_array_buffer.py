from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from skfolio.utils._array_buffer import _ArrayBuffer


class TestArrayBufferInit:
    def test_empty_init(self):
        g = _ArrayBuffer()
        assert len(g) == 0
        assert g.array is None

    def test_repr_empty(self):
        g = _ArrayBuffer()
        assert repr(g) == "_ArrayBuffer(empty)"

    def test_repr_with_data(self):
        g = _ArrayBuffer()
        g.append(np.zeros((3, 4)))
        r = repr(g)
        assert "len=3" in r
        assert "dtype=float64" in r


class TestArrayBufferAppend:
    def test_single_row_appends(self):
        g = _ArrayBuffer()
        for i in range(5):
            g.append(np.array([[float(i)]]))
        assert len(g) == 5
        npt.assert_array_equal(g.array, np.arange(5.0).reshape(5, 1))

    def test_batch_append(self):
        g = _ArrayBuffer()
        g.append(np.arange(12.0).reshape(3, 4))
        assert len(g) == 3
        npt.assert_array_equal(g.array, np.arange(12.0).reshape(3, 4))

    def test_mixed_append_sizes(self):
        g = _ArrayBuffer()
        g.append(np.array([[1.0, 2.0]]))
        g.append(np.array([[3.0, 4.0], [5.0, 6.0]]))
        g.append(np.array([[7.0, 8.0]]))
        assert len(g) == 4
        expected = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
        npt.assert_array_equal(g.array, expected)

    def test_geometric_growth(self):
        g = _ArrayBuffer()
        g.append(np.zeros((2, 3)))
        initial_capacity = g._buffer.shape[0]
        assert initial_capacity == 2
        g.append(np.zeros((1, 3)))
        assert g._buffer.shape[0] >= 3
        assert g._buffer.shape[0] == initial_capacity * 2

    def test_3d_append(self):
        g = _ArrayBuffer()
        g.append(np.ones((1, 4, 5)))
        g.append(np.ones((2, 4, 5)) * 2)
        assert len(g) == 3
        assert g.array.shape == (3, 4, 5)
        npt.assert_array_equal(g.array[0], 1.0)
        npt.assert_array_equal(g.array[1:], 2.0)

    def test_dtype_preservation(self):
        g = _ArrayBuffer()
        g.append(np.array([[1, 2]], dtype=np.float32))
        g.append(np.array([[3, 4]], dtype=np.float32))
        assert g.array.dtype == np.float32

    def test_many_appends_correctness(self):
        g = _ArrayBuffer()
        rows = [np.array([[float(i)]]) for i in range(100)]
        for r in rows:
            g.append(r)
        expected = np.arange(100.0).reshape(100, 1)
        npt.assert_array_equal(g.array, expected)

    def test_shape_mismatch_raises(self):
        g = _ArrayBuffer()
        g.append(np.zeros((2, 3)))
        with pytest.raises(ValueError, match="Shape mismatch"):
            g.append(np.zeros((2, 5)))

    def test_dtype_mismatch_raises(self):
        g = _ArrayBuffer()
        g.append(np.array([[1, 2]], dtype=np.int64))
        with pytest.raises(ValueError, match="dtype mismatch"):
            g.append(np.array([[1.5, 2.5]], dtype=np.float64))

    def test_1d_input_promoted_to_2d(self):
        g = _ArrayBuffer()
        g.append(np.array([1.0, 2.0, 3.0]))
        assert g.array.shape == (1, 3)
        npt.assert_array_equal(g.array, [[1.0, 2.0, 3.0]])


class TestArrayBufferView:
    def test_array_is_view(self):
        g = _ArrayBuffer()
        g.append(np.array([[1.0, 2.0]]))
        g.append(np.array([[3.0, 4.0]]))
        view = g.array
        assert view.base is g._buffer

    def test_view_is_read_only(self):
        g = _ArrayBuffer()
        g.append(np.array([[1.0, 2.0]]))
        view = g.array
        assert not view.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            view[0, 0] = 999.0

    def test_view_unaffected_by_subsequent_append(self):
        g = _ArrayBuffer()
        g.append(np.array([[1.0]]))
        view1 = g.array.copy()
        g.append(np.array([[2.0]]))
        npt.assert_array_equal(view1, np.array([[1.0]]))

    def test_view_shape(self):
        g = _ArrayBuffer()
        g.append(np.zeros((3, 5, 7)))
        assert g.array.shape == (3, 5, 7)
        g.append(np.zeros((2, 5, 7)))
        assert g.array.shape == (5, 5, 7)


class TestArrayBufferTruncate:
    def test_truncate_to_last(self):
        g = _ArrayBuffer()
        for i in range(10):
            g.append(np.array([[float(i)]]))
        g.truncate_to_last(3)
        assert len(g) == 3
        npt.assert_array_equal(g.array, np.array([[7], [8], [9]], dtype=float))

    def test_truncate_no_op_when_shorter(self):
        g = _ArrayBuffer()
        g.append(np.array([[1.0], [2.0]]))
        g.truncate_to_last(5)
        assert len(g) == 2
        npt.assert_array_equal(g.array, np.array([[1], [2]], dtype=float))

    def test_truncate_exact_length(self):
        g = _ArrayBuffer()
        g.append(np.array([[1.0], [2.0], [3.0]]))
        g.truncate_to_last(3)
        assert len(g) == 3
        npt.assert_array_equal(g.array, np.array([[1], [2], [3]], dtype=float))

    def test_truncate_then_append(self):
        g = _ArrayBuffer()
        for i in range(10):
            g.append(np.array([[float(i)]]))
        g.truncate_to_last(2)
        g.append(np.array([[99.0]]))
        assert len(g) == 3
        npt.assert_array_equal(g.array, np.array([[8], [9], [99]], dtype=float))

    def test_truncate_to_one(self):
        g = _ArrayBuffer()
        for i in range(5):
            g.append(np.array([[float(i)]]))
        g.truncate_to_last(1)
        assert len(g) == 1
        npt.assert_array_equal(g.array, np.array([[4.0]]))

    def test_truncate_zero_raises(self):
        g = _ArrayBuffer()
        g.append(np.array([[1.0]]))
        with pytest.raises(ValueError, match="positive"):
            g.truncate_to_last(0)

    def test_truncate_negative_raises(self):
        g = _ArrayBuffer()
        g.append(np.array([[1.0]]))
        with pytest.raises(ValueError, match="positive"):
            g.truncate_to_last(-1)


class TestArrayBufferClear:
    def test_clear_releases_memory(self):
        g = _ArrayBuffer()
        g.append(np.zeros((10, 3)))
        g.clear()
        assert len(g) == 0
        assert g.array is None

    def test_clear_then_append(self):
        g = _ArrayBuffer()
        g.append(np.array([[1.0, 2.0]]))
        g.clear()
        g.append(np.array([[3.0, 4.0]]))
        assert len(g) == 1
        npt.assert_array_equal(g.array, [[3.0, 4.0]])


class TestArrayBufferEdgeCases:
    def test_append_empty_2d_is_no_op(self):
        g = _ArrayBuffer()
        g.append(np.array([[1.0, 2.0]]))
        g.append(np.empty((0, 2)))
        assert len(g) == 1
        npt.assert_array_equal(g.array, np.array([[1.0, 2.0]]))

    def test_append_empty_1d_is_no_op(self):
        g = _ArrayBuffer()
        g.append(np.array([[1.0, 2.0]]))
        g.append(np.array([]))
        assert len(g) == 1
        npt.assert_array_equal(g.array, np.array([[1.0, 2.0]]))

    def test_append_empty_1d_on_empty_buffer(self):
        g = _ArrayBuffer()
        g.append(np.array([]))
        assert len(g) == 0
        assert g.array is None

    def test_append_zero_width_rows(self):
        g = _ArrayBuffer()
        g.append(np.empty((2, 0)))
        assert len(g) == 2
        assert g.array.shape == (2, 0)
        g.append(np.empty((3, 0)))
        assert len(g) == 5
        assert g.array.shape == (5, 0)

    def test_first_append_is_exact_fit(self):
        g = _ArrayBuffer()
        data = np.zeros((100, 5))
        g.append(data)
        assert g._buffer.shape[0] == 100
