"""Observation normalization and persistence shared by panels and graph series."""

import json
import pickle
from datetime import datetime, timedelta, timezone, tzinfo
from itertools import repeat
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from skfolio.containers import AssetPanel, Graph, GraphSeries, concat
from skfolio.containers._io import (
    _decode_observations,
    _encode_observations,
    _read_manifest,
)
from skfolio.containers._utils import _as_observation_array


@pytest.fixture(params=[AssetPanel, GraphSeries])
def make_container(request):
    def make(observations):
        if request.param is AssetPanel:
            return AssetPanel(
                observations=observations,
                asset_names=["A"],
                fields={"returns": np.zeros((len(observations), 1))},
            )
        graph = Graph([[0, 1], [0, 0]], nodes=["A", "Supplier"])
        return GraphSeries(repeat(graph, len(observations)), observations=observations)

    return make


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("zone", ["America/New_York", "Asia/Tokyo", "UTC"])
@pytest.mark.parametrize("input_type", ["index", "series", "array", "list", "python"])
def test_aware_observation_inputs(zone, input_type):
    dates = pd.date_range("2024-03-09", periods=3, tz=zone)
    values = {
        "index": dates,
        "series": pd.Series(dates),
        "array": dates.to_numpy(),
        "list": list(dates),
        "python": dates.to_pydatetime(),
    }[input_type]
    observations = _as_observation_array(values)
    assert observations.dtype == object
    assert pd.DatetimeIndex(observations).equals(dates)


@pytest.mark.parametrize(
    "missing", [pd.NaT, np.datetime64("NaT", "ns"), np.datetime64("NaT", "us"), None]
)
def test_timezone_aware_dates_with_missing_values(make_container, missing):
    date = pd.Timestamp("2024-01-01 09:00:00.123456789", tz="America/New_York")
    container = make_container([date, missing])
    assert pd.DatetimeIndex(container.observations).equals(
        pd.DatetimeIndex([date, pd.NaT])
    )


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
@pytest.mark.parametrize("missing", [False, True])
def test_empty_or_missing_aware_dates_have_unit(unit, missing):
    dates = pd.DatetimeIndex([pd.NaT] if missing else [], tz="Asia/Tokyo").as_unit(unit)
    observations = _as_observation_array(dates)
    assert observations.dtype == np.dtype(f"datetime64[{unit}]")
    assert pd.Index(observations).isna().all()


@pytest.mark.parametrize(
    "values",
    [np.array([], dtype=object), [pd.NaT], np.array([], dtype="datetime64")],
)
def test_dates_without_known_unit_use_nanoseconds(values):
    observations = _as_observation_array(values)
    assert observations.dtype == np.dtype("datetime64[ns]")
    assert pd.Index(observations).isna().all()


@pytest.mark.parametrize("unit", ["D", "s", "ms", "us", "ns"])
def test_naive_datetime_units_unchanged(unit):
    dates = np.array(["2024-01-01", "NaT"], dtype=f"datetime64[{unit}]")
    assert _as_observation_array(dates) is dates


@pytest.mark.parametrize(
    "other", [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02", tz="UTC")]
)
def test_mixed_timezones_are_rejected(make_container, other):
    with pytest.raises(ValueError, match="single timezone"):
        make_container([pd.Timestamp("2024-01-01", tz="America/New_York"), other])


@pytest.mark.filterwarnings("error")
def test_container_operations_preserve_timezone(make_container):
    dates = pd.date_range("2024-03-08", periods=5, tz="America/New_York")
    container = make_container(dates)
    for result in [
        container.copy(),
        container.copy(deep=True),
        pickle.loads(pickle.dumps(container)),
    ]:
        assert pd.DatetimeIndex(result.observations).equals(dates)
        if isinstance(result, GraphSeries):
            assert result[0] is result[-1]
    assert pd.DatetimeIndex(container[[3, 1]].observations).equals(dates[[3, 1]])
    assert pd.DatetimeIndex(concat([container[:2], container[2:]]).observations).equals(
        dates
    )
    if isinstance(container, AssetPanel):
        for selected in [dates[1], dates[1].to_pydatetime(), dates[[1, 3]]]:
            result = container.sel(observations=selected)
            expected = (
                pd.DatetimeIndex([selected])
                if pd.api.types.is_scalar(selected)
                else selected
            )
            assert pd.DatetimeIndex(result.observations).equals(expected)
        assert pd.DatetimeIndex(container[1:3].to_panel().observations).equals(
            dates[1:3]
        )


def test_missing_date_slice_and_concat(make_container, tmp_path):
    dates = pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="UTC"), pd.NaT])
    container = make_container(dates)
    missing = container[1:]
    if isinstance(container, AssetPanel):
        missing = missing.to_panel()
    missing = missing.copy()
    missing.save(tmp_path / "missing")
    restored = type(container).load(tmp_path / "missing")
    assert restored.observations.dtype == np.dtype("datetime64[ns]")
    combined = concat([container[:1], restored])
    assert pd.DatetimeIndex(combined.observations).equals(dates)


def test_concat_rejects_different_timezones(make_container):
    first = make_container(pd.date_range("2024-01-01", periods=2, tz="UTC"))
    second = make_container(pd.date_range("2024-01-03", periods=2, tz="Asia/Tokyo"))
    with pytest.raises(ValueError, match="single timezone"):
        concat([first, second])


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "zone", ["America/New_York", "Asia/Tokyo", "UTC", "+05:30", "-03:30"]
)
def test_timezone_round_trip(make_container, tmp_path, zone):
    # Both occurrences of 01:30 during the New York clock change remain distinct.
    dates = pd.date_range(
        "2024-11-03 05:30:00.123456789", periods=3, freq="h", tz="UTC"
    ).tz_convert(zone)
    container = make_container(dates)
    container.save(tmp_path / "saved")
    metadata = json.loads((tmp_path / "saved" / "_metadata.json").read_text())
    assert metadata["version"] == 2
    assert metadata["observations_timezone"] == zone
    stored_dates = np.load(tmp_path / "saved" / "observations.npy", allow_pickle=False)
    np.testing.assert_array_equal(
        stored_dates, dates.tz_convert("UTC").tz_localize(None).to_numpy()
    )
    loaded = type(container).load(tmp_path / "saved")
    assert pd.DatetimeIndex(loaded.observations).equals(dates)
    with pytest.raises(ValueError, match="format version 2"):
        _read_manifest(
            tmp_path / "saved", container=type(container).__name__, max_version=1
        )


@pytest.mark.parametrize("all_missing", [False, True])
def test_missing_dates_and_empty_slices_round_trip(
    make_container, tmp_path, all_missing
):
    dates = pd.DatetimeIndex(
        [pd.NaT] if all_missing else [pd.Timestamp("2024-01-01", tz="UTC"), pd.NaT]
    )
    container = make_container(dates)
    container.save(tmp_path / "saved")
    loaded = type(container).load(tmp_path / "saved")
    assert pd.DatetimeIndex(loaded.observations).equals(
        pd.DatetimeIndex(container.observations)
    )
    empty = container[:0]
    if isinstance(container, AssetPanel):
        empty = empty.to_panel()
    empty.save(tmp_path / "empty")
    assert len(type(container).load(tmp_path / "empty")) == 0


@pytest.mark.parametrize(
    "observations",
    [
        np.arange(3),
        ["first", "second", "third"],
        pd.date_range("2024-01-01", periods=3),
    ],
)
def test_ordinary_observations_keep_version_one(make_container, tmp_path, observations):
    container = make_container(observations)
    container.save(tmp_path / "saved")
    metadata_path = tmp_path / "saved" / "_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["version"] == 1
    assert "observations_timezone" not in metadata
    if isinstance(container, AssetPanel):
        del metadata["container"]
        metadata_path.write_text(json.dumps(metadata))
    loaded = type(container).load(tmp_path / "saved")
    np.testing.assert_array_equal(loaded.observations, container.observations)


@pytest.mark.parametrize("minutes", [0, 330, -210])
def test_fixed_offset_with_custom_name(minutes):
    zone = timezone(timedelta(minutes=minutes), name="Custom name")
    dates = pd.date_range("2024-01-01", periods=2, tz=zone)
    values, zone_name = _encode_observations(observations=_as_observation_array(dates))
    restored = _decode_observations(observations=values, timezone=zone_name)
    np.testing.assert_array_equal(restored, dates.to_numpy())


def test_dateutil_named_timezone():
    from dateutil.tz import gettz

    zone = gettz("America/New_York")
    try:
        dates = pd.date_range("2024-03-09", periods=3, tz=zone)
    except AttributeError:
        pytest.skip(
            "This pandas/dateutil combination cannot construct a named dateutil index"
        )
    observations = _as_observation_array(dates)
    assert pd.DatetimeIndex(observations).equals(dates)
    if getattr(zone, "key", None) is None:
        with pytest.raises(ValueError, match="named timezone"):
            _encode_observations(observations=observations)
    else:
        values, zone_name = _encode_observations(observations=observations)
        assert zone_name == "America/New_York"
        np.testing.assert_array_equal(
            _decode_observations(observations=values, timezone=zone_name), observations
        )


@pytest.mark.parametrize("fixed", [False, True])
def test_pytz_timezones(fixed):
    pytz = pytest.importorskip("pytz")
    zone = pytz.FixedOffset(330) if fixed else pytz.timezone("America/New_York")
    dates = pd.date_range("2024-03-09", periods=3, tz=zone)
    observations = _as_observation_array(dates)
    values, zone_name = _encode_observations(observations=observations)
    np.testing.assert_array_equal(
        _decode_observations(observations=values, timezone=zone_name), observations
    )


class InvalidNamedTimezone(tzinfo):
    key = "Unknown/Timezone"

    def utcoffset(self, dt):
        return timedelta(hours=1)

    def dst(self, dt):
        return timedelta(0)


@pytest.mark.parametrize(
    "zone", [InvalidNamedTimezone(), timezone(timedelta(seconds=30))]
)
def test_unsupported_timezone_preserves_existing_save(make_container, tmp_path, zone):
    saved = make_container(pd.date_range("2024-01-01", periods=2))
    saved.save(tmp_path / "saved")
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in (tmp_path / "saved").rglob("*")
        if path.is_file()
    }
    container = make_container([datetime(2024, 1, 1, tzinfo=zone)])
    with pytest.raises(ValueError, match="timezone"):
        container.save(tmp_path / "saved", overwrite=True)
    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in (tmp_path / "saved").rglob("*")
        if path.is_file()
    }
    assert before == after


@pytest.mark.parametrize("zone", ["Unknown/Timezone", ["UTC"]])
def test_invalid_saved_timezone(make_container, tmp_path, zone):
    container = make_container(
        pd.date_range("2024-01-01", periods=2, tz=ZoneInfo("UTC"))
    )
    container.save(tmp_path / "saved")
    path = tmp_path / "saved" / "_metadata.json"
    metadata = json.loads(path.read_text())
    metadata["observations_timezone"] = zone
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="observations timezone"):
        type(container).load(tmp_path / "saved")


def test_saved_timezone_requires_datetime_values(make_container, tmp_path):
    container = make_container(pd.date_range("2024-01-01", periods=2, tz="UTC"))
    container.save(tmp_path / "saved")
    np.save(tmp_path / "saved" / "observations.npy", np.arange(2))
    with pytest.raises(
        ValueError, match="timezone requires a string and datetime values"
    ):
        type(container).load(tmp_path / "saved")
