import datetime
import io
import os
from typing import List

import httpx
import pandas as pd


async def get_grib_idx(date: datetime, forecast_hour: int):
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{date.year:04d}{date.month:02d}{date.day:02d}/conus/hrrr.t{date.hour:02d}z.wrfsubhf{str(forecast_hour).zfill(2)}.grib2.idx"
        )
    return r


def parse_to_df(r):
    """
    Pass the grib.idx bytes response into a pandas dataframe.
    """
    return pd.read_csv(io.StringIO(r.text), sep=":", index_col=0, header=None).dropna(
        axis=1, how="all"
    )


def get_byte_locs(dloc: pd.DataFrame, variable: str, level: str):
    """
    var is in 3rd column, 4th is level
    """
    return dloc[(dloc[3] == variable) & (dloc[4] == level)]


def get_forecast_metadata(dlocs: pd.DataFrame):
    """
    get level, variable, and strip time/type metadata in 5th column
    """
    lvl = dlocs[4].unique()
    assert len(lvl) == 1
    var = dlocs[3].unique()
    assert len(var) == 1
    meta = [(var[0], lvl[0], x) for x in dlocs[5].str.split(" ", 1).to_list()]
    return meta


def get_byte_ranges(dlocs: pd.DataFrame, dparsed: pd.DataFrame):
    """
    byte range is that row index's first column and ends with the next row's byte start.
    """
    return [
        (dparsed.iloc[r[0]][1], dparsed.iloc[r[0] + 1][1]) for r in dlocs.iterrows()
    ]


def build_queries(r, out_dir: str, cfg: List):
    """
    Create curl requests and save a unique filename based on metadata.
    """
    frqst = r._request.url.path
    queries = [
        f"curl -o {out_dir}{frqst.replace('/', '').replace('.grib2.idx', '')}_{x[0][0].strip()}_{x[0][1].strip()}_{''.join(x[0][2]).replace(' ','_').strip()}.grib2 --range {x[1][0]}-{x[1][1]} https://noaa-hrrr-bdp-pds.s3.amazonaws.com{frqst.replace('.idx', '')}"
        for x in cfg
    ]
    return queries


if __name__ == "__main__":
    r = await get_grib_idx(
        date=datetime.datetime.utcnow() - datetime.timedelta(days=1), forecast_hour=3
    )
    if r.status_code == 200:
        dparsed = parse_to_df(r)
        dlocs = get_byte_locs(dparsed, variable="PRATE", level="surface")
        dmeta = get_forecast_metadata(dlocs)
        dranges = get_byte_ranges(dlocs=dlocs, dparsed=dparsed)
        queries = build_queries(r, out_dir="/tmp/", cfg=list(zip(dmeta, dranges)))
        # os.system(queries[0])
        # [os.system(x) for x in queries]
