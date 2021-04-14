from fastapi import FastAPI
from pydantic import BaseModel
from gribflow.serve import encode_array, decode_array, from_epoch, get_forecast_gribs, round_datetime, get_tws, to_epoch, interpolate, resize, get_file_valid_times
from gribflow.io import get_models, get_gribs, natural_keys
from gribflow.grib import _read_headers, _read_vals
import json
import datetime
import itertools
import numpy as np
from typing import Optional
import re
import shutil
import tempfile

app = FastAPI()

@app.get("/")
def read_root():
    return get_models()


@app.get("/data/{model}/{product}/{file}/{variable}/{level}/{forecast}/{epoch}")
async def get_data(epoch: int, model: str, product: str, file: str, variable: str, level: str, forecast: str, xy: Optional[str] = None, method: Optional[str] = 'bilinear'):
    # return {"epoch": epoch, "model": model, "product": product, "file": file, "variable": variable, "level": level, "forecast": forecast}

    cfg = get_models()[model.lower()]
    cfg = cfg['products'][product.lower()]['files'][file.lower()]
    # cfg = get_models()['hrrr'.lower()]
    # cfg = cfg['products']['conus'.lower()]['files']['wrfsubhf'.lower()]
    mytime = from_epoch(epoch)
    candidates = get_file_valid_times(mytime = mytime, cfg = cfg)

    rng = np.random.default_rng(123)

    rng.shuffle(candidates['first'], axis=0)
    rng.shuffle(candidates['last'], axis=0)
    best_lower_delta = -np.inf
    best_upper_delta = np.inf

    tempdir = tempfile.mkdtemp()

    for i in range(0, 5):
        try:
            # the datetime of the model + the forecast hour
            file_queries = (candidates['first'][i,0], int(candidates['first'][i,1].total_seconds() / (60 * 60))), (candidates['last'][i,0], int(candidates['last'][i,1].total_seconds() / (60 * 60)))
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
            hdrs_valid_times = [datetime.datetime.strptime(f"{x[5]} {x[6]:02d}", "%Y%m%d %H%M") for x in hdrs]
            print(hdrs_valid_times)
            # sort on delta
            lower = min([ x for x in hdrs_valid_times if x < mytime.replace(tzinfo=None)], key=lambda x:abs(x-mytime.replace(tzinfo=None)))
            lower_delta = (lower - mytime.replace(tzinfo=None)).total_seconds()
            upper = min([ x for x in hdrs_valid_times if x >= mytime.replace(tzinfo=None)], key=lambda x:abs(x-mytime.replace(tzinfo=None)))
            upper_delta = (upper - mytime.replace(tzinfo=None)).total_seconds()

            if best_lower_delta < lower_delta < 0:
                best_lower = lower
                best_lower_delta = lower_delta

            if best_upper_delta > upper_delta > 0:
                best_upper = upper
                best_upper_delta = upper_delta

            print(f"upper {upper} upper_delta {upper_delta} best_upper {best_upper} best_upper_delta {best_upper_delta}")

        except Exception as e:
            print(e)
            continue

        break
        # print(f"lower {lower} lower_delta {lower_delta} best_lower {best_lower} best_lower_delta {best_lower_delta}")

    shutil.rmtree(tempdir)
    return [best_lower, lower, best_upper, upper ]

    '''
    # get the matching grib files
    filelower = grib_files[hdrs_valid_times.index(lower)]
    fileupper = grib_files[hdrs_valid_times.index(upper)]

    target_tw = get_tws(target = epoch, start = to_epoch(lower), end = to_epoch(upper)) 

    x1 = _read_vals(filelower)
    x2 = _read_vals(fileupper)

    hat = interpolate(ar1=x1, ar2=x2, tws=[target_tw], flow_ar=None)

    if xy:
        # parse size if provided
        x, y = re.findall(r'\d+', xy)
        x, y = int(x), int(y)
        hat = resize(hat, shape=(y, x), method=method)

    return {'start_file': filelower, 'end_file': fileupper, 'time_weight': target_tw, 'array':  encode_array(np.array(hat))}
    '''