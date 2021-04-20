import asyncio
import base64
import calendar
import datetime
import itertools
import math
from functools import partial
from typing import Optional

import blosc
import jax
import jax.numpy as jnp
import numpy as np

from gribflow.flow import calc_opt_flow, interpolate_frames, np_to_gray
from gribflow.io import get_gribs


def to_epoch(x: datetime):
    """
    UTC datetime to epoch in [ms]
    """
    if x.tzinfo:
        x = x.astimezone(datetime.timezone.utc)
    return calendar.timegm(x.timetuple()) * 1000


def from_epoch(x: int):
    """
    UTC epoch in [ms] to datetime
    """
    return datetime.datetime.utcfromtimestamp(x / 1000).astimezone(
        datetime.timezone.utc
    )


def round_datetime(dt, secperiod=15 * 60, method="ceil"):
    """
    Take datetime and round up or down to nearest second period
    """
    tstmp = dt.timestamp()
    if "ceil" in method.lower():
        return (
            datetime.datetime.fromtimestamp(math.ceil(tstmp / secperiod) * secperiod)
            .astimezone()
            .astimezone(datetime.timezone.utc)
        )
    else:
        return (
            datetime.datetime.fromtimestamp(math.floor(tstmp / secperiod) * secperiod)
            .astimezone()
            .astimezone(datetime.timezone.utc)
        )


def get_tws(target, start, end):
    """
    Given a start and end epoch integer, calculate where the target time weight slice would be
        ex:
        get_tws(target = 1617632280000, start = 1617632100000, end = 1617633000000)
    """
    assert target >= start
    assert target <= end
    epoch_delta = end - start
    return (target - start) / epoch_delta


def closest_time_point(array, values, n, method="floor"):
    # time delta
    dist = np.subtract.outer(array[:, -1], values)
    if method == "floor":
        # if floor we only keep times less than or equal to the floor date
        mtch_locs = array[dist <= datetime.timedelta(seconds=0)]
    elif method == "ceil":
        # if ceil we only keep times greater than or equal to the ceil date
        mtch_locs = array[dist >= datetime.timedelta(seconds=0)]
    dist = np.abs(np.subtract.outer(mtch_locs[:, -1], values))
    return mtch_locs[np.argpartition(dist, n)[:n]]


def encode_array(array, compressor=partial(blosc.compress, cname="lz4")):
    """
    Compressor numpy array to json-safe string
    """
    cdata = base64.urlsafe_b64encode(
        compressor(np.array(array).astype(np.float64))
    ).decode("utf-8")
    return cdata


def decode_array(cdata, shape, compressor=blosc.decompress):
    """
    Decode string to bytes and uncompress to numpy array
    """
    data = np.frombuffer(
        compressor(base64.urlsafe_b64decode(cdata)), dtype=np.float64
    ).reshape(shape)
    return data


def interpolate(ar1, ar2, tws, flow_ar=None):
    """
    Interpolate frame for a requested slice between the two.
        ex:
        interpolate(ar1, ar2, tws=[.2, .5])
    """
    if flow_ar is None:
        armax = np.nanmax([ar1, ar2])
        armin = np.nanmin([ar1, ar2])
        gray1 = np_to_gray(ar1, floor=armin, ceil=armax)
        gray2 = np_to_gray(ar2, floor=armin, ceil=armax)
        flow_ar = calc_opt_flow(gray1, gray2)
    hat = interpolate_frames(ar1, ar2, flow_ar, tws=tws)
    return jnp.stack(hat).squeeze(axis=0)


def resize(jar, shape, method="bicubic"):
    return jax.image.resize(jar, shape=shape, method=method)


def get_file_valid_times(
    mytime: datetime, cfg: dict, n_run_searches: Optional[int] = 96
):
    # round provided time to nearest run file.
    cdatetime = datetime.datetime.utcnow()
    mytime_floor = round_datetime(
        min([cdatetime.replace(tzinfo=None), mytime.replace(tzinfo=None)]), secperiod=cfg["run_hour_delta"] * 60 * 60, method="floor"
    ).replace(tzinfo=None)
    # go back up to n_run_searches model runs
    file_range = [
        mytime_floor - datetime.timedelta(hours=cfg["run_hour_delta"] * x)
        for x in range(0, n_run_searches)
    ]
    # product of mydate, the within file time steps, and all the forecast hours for the run
    # ['model_run', 'forecast_time', 'within_file_time']
    cart = np.array(
        list(
            itertools.product(
                file_range,
                [
                    datetime.timedelta(hours=x)
                    for x in range(0, cfg["max_hour_fcst"] + 1)
                ],
                cfg["within_file_timesteps"],
            )
        )
    )
    # compute valid time
    cart_valid = (
        cart[:, 0]
        + cart[:, 1]
        - datetime.timedelta(hours=cfg["fcst_hour_delta"])
        + cart[:, 2]
    )
    # delta to mytime
    delta = cart_valid - mytime.replace(tzinfo=None)
    cart_valid = np.concatenate(
        [cart, cart_valid.reshape(-1, 1), delta.reshape(-1, 1)], axis=1
    )
    # sort on delta
    cart_valid = cart_valid[cart_valid[:, -1].argsort()]
    # find closest points to time and split into before and after
    idx = np.searchsorted(cart_valid[:, -2], mytime.replace(tzinfo=None))
    candidates = {"first": cart_valid[0:idx], "last": cart_valid[idx:]}
    # sort for best available. minimum delta and smaller forecast time
    ind = np.lexsort(
        (candidates["last"][:, 0], candidates["last"][:, 1], candidates["last"][:, -1])
    )
    candidates["last"] = candidates["last"][ind]
    # descending order for delta of first candidates
    ind = np.lexsort(
        (
            candidates["first"][:, 0],
            candidates["first"][:, 1],
            -candidates["first"][:, -1],
        )
    )
    candidates["first"] = candidates["first"][ind]
    return candidates



async def get_forecast_gribs(
    time_queries: dict,
    model: str,
    product: str,
    file: str,
    variable: str,
    level: str,
    forecast: str,
    out_dir: str,
):
    tasks = [
        asyncio.create_task(
            get_gribs(
                timestamp=datetime.datetime.strftime(x[0], "%Y-%m-%d %H:%M:%S"),
                forecast_hour=int(x[1]),
                model=model,
                product=product,
                file=file,
                variable=variable,
                level=level,
                forecast=forecast,
                out_dir=out_dir,
            )
        )
        for x in time_queries
    ]
    results = await asyncio.gather(*tasks)
    return results
