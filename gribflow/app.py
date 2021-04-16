import datetime
import glob
import re
import shutil
import tempfile
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException

from gribflow.flow import inpaint
from gribflow.grib import _read_headers, _read_vals
from gribflow.io import get_models
from gribflow.serve import (encode_array, from_epoch,
                            get_file_valid_times, get_forecast_gribs, get_tws,
                            interpolate, resize, to_epoch)

# instantiate app
app = FastAPI()


@app.get("/")
def read_root():
    return get_models()


@app.get("/data/{model}/{product}/{file}/{variable}/{level}/{forecast}/{epoch}")
async def get_data(
    epoch: int,
    model: str,
    product: str,
    file: str,
    variable: str,
    level: str,
    forecast: str,
    xy: Optional[str] = None,
    method: Optional[str] = "bicubic",
):

    cfg = get_models()[model.lower()]
    cfg = cfg["products"][product.lower()]["files"][file.lower()]
    mytime = from_epoch(epoch)

    # grab candidate files
    candidates = get_file_valid_times(mytime=mytime, cfg=cfg, n_run_searches=96)
    max_range = min(len(candidates["first"]), len(candidates["last"]))

    # initialize placeholders as we attempt to download most relevant files.
    best_lower_delta = -np.inf
    best_lower = -np.inf
    best_upper_delta = np.inf
    best_upper = np.inf

    tempdir = tempfile.mkdtemp()

    success = None
    for i in range(0, 3):
        try:
            # the datetime of the model + the forecast hour
            file_queries = (
                candidates["first"][i, 0],
                int(candidates["first"][i, 1].total_seconds() / (60 * 60)),
            ), (
                candidates["last"][i, 0],
                int(candidates["last"][i, 1].total_seconds() / (60 * 60)),
            )
            file_queries = list(set(file_queries))

            grib_files = await get_forecast_gribs(
                time_queries=file_queries,
                model=model,
                variable=variable,
                product=product,
                file=file,
                level=level,
                forecast=forecast,
                out_dir=tempdir,
            )

            # combine if multiple lists
            if any(isinstance(x, list) for x in grib_files):
                grib_files = sum(grib_files, [])
            # get hdrs
            hdrs = [_read_headers(x) for x in grib_files]
            hdrs_valid_times = [
                datetime.datetime.strptime(f"{x[5]} {x[6]:02d}", "%Y%m%d %H%M")
                for x in hdrs
            ]
            # breaks here since deltas would be same!
            # sort on delta
            lower = min(
                [x for x in hdrs_valid_times if x < mytime.replace(tzinfo=None)],
                key=lambda x: abs(x - mytime.replace(tzinfo=None)),
            )
            lower_delta = (lower - mytime.replace(tzinfo=None)).total_seconds()
            upper = min(
                [x for x in hdrs_valid_times if x >= mytime.replace(tzinfo=None)],
                key=lambda x: abs(x - mytime.replace(tzinfo=None)),
            )
            upper_delta = (upper - mytime.replace(tzinfo=None)).total_seconds()
            if best_lower_delta < lower_delta <= 0:
                best_lower = lower
                best_lower_delta = lower_delta

            if best_upper_delta > upper_delta >= 0:
                best_upper = upper
                best_upper_delta = upper_delta

        except Exception as e:
            print(e)
            continue
        
        success = True
        break

    if success is None:
        raise HTTPException(status_code=404, detail="Data not found")

    # now get those matching grib files
    filelower = grib_files[hdrs_valid_times.index(best_lower)]
    fileupper = grib_files[hdrs_valid_times.index(best_upper)]

    x1 = _read_vals(filelower)
    x2 = _read_vals(fileupper)

    if epoch == to_epoch(best_lower):
        hat = x1
    
    elif epoch == to_epoch(best_upper):
        hat = x2

    else:
        target_tw = get_tws(
            target=epoch, start=to_epoch(best_lower), end=to_epoch(best_upper)
        )
        hat = interpolate(ar1=x1, ar2=x2, tws=[target_tw], flow_ar=None)

    if xy:
        # parse size if provided and interpolate
        x, y = re.findall(r"\d+", xy)
        x, y = int(x), int(y)
        # fillnans as jax does not like interpolating with them
        hat = resize(np.nan_to_num(np.array(hat)), shape=(y, x), method=method)

    shutil.rmtree(tempdir)

    hat = np.array(hat)

    return {
        "start_file": filelower,
        "end_file": fileupper,
        "timestamp": mytime,
        "epoch": epoch,
        "shape": hat.shape,
        "array": encode_array(hat),
    }
