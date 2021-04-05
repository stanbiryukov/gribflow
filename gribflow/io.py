import argparse
import asyncio
import datetime
import io
import shutil
import urllib
from typing import List

import aiohttp
import pandas as pd


async def fetch(session, url):
    """
    Create aiohttp request.
    """
    async with session.get(url) as response:
        if response.status != 200:
            response.raise_for_status()
        return await response.text()


async def fetch_all(session, urls):
    """
    Create aiohttp requests for multiple urls.
    """
    tasks = []
    for url in urls:
        task = asyncio.create_task(fetch(session, url))
        tasks.append(task)
    results = await asyncio.gather(*tasks)
    return results


def create_grib_idx_url_path(timestamp: datetime, forecast_hour: int):
    """
    Given a date and forecast hour return the expected grib.idx filepath on the open data S3 bucket for HRRR `wrfsubh` file.
    """
    return f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{timestamp.year:04d}{timestamp.month:02d}{timestamp.day:02d}/conus/hrrr.t{timestamp.hour:02d}z.wrfsubhf{str(forecast_hour).zfill(2)}.grib2.idx"


def parse_to_df(r):
    """
    Pass the grib.idx bytes response into a pandas dataframe.
    """
    return pd.read_csv(io.StringIO(r), sep=":", index_col=0, header=None).dropna(
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
        (dparsed.loc[r][1], dparsed.loc[r + 1][1]) for r in dlocs.index
    ]


def download_grib_chunk(url: str, path: str, _range=None):
    # print(f"Fetching: {url} with byte header: {_range} and saving to: {path}")
    req = urllib.request.Request(url, method="GET")
    if _range:
        req.add_header("Range", _range)  # 'bytes=b0-b1)'
    with urllib.request.urlopen(req) as response:
        with open(path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)


def download_files(idx_url, out_dir: str, cfg: List):
    """
    Download the files and save a unique filename based on metadata collected.
    """
    [
        download_grib_chunk(
            url=idx_url.replace(".idx", ""),
            path=f"{out_dir}/{'_'.join(idx_url.rsplit('/')[-3:]).replace('.grib2.idx', '')}_{x[0][0].strip()}_{x[0][1].strip()}_{''.join(x[0][2]).replace(' ','_').strip()}.grib2",
            _range=f"bytes={x[1][0]}-{x[1][1]}",
        )
        for x in cfg
    ]


async def main(args):
    idx_url = create_grib_idx_url_path(
        timestamp=pd.to_datetime(args.timestamp, utc=True), forecast_hour=args.forecast_hour
    )
    async with aiohttp.ClientSession() as session:
        r = await fetch(session, idx_url)

    dparsed = parse_to_df(r)
    dlocs = get_byte_locs(dparsed, variable=args.variable, level=args.level)
    dmeta = get_forecast_metadata(dlocs)
    dranges = get_byte_ranges(dlocs=dlocs, dparsed=dparsed)
    download_files(idx_url=idx_url, out_dir=args.out_dir, cfg=list(zip(dmeta, dranges)))


if __name__ == "__main__":
    '''
    Fetch subsets of HRRR sub-hourly forecast outputs.
        ex: # python get_gribs.py -timestamp '2021-03-30 03:15:00Z' -forecast_hour 6 -variable 'PRATE' -level 'surface' -out_dir '/tmp'
    '''
    parser = argparse.ArgumentParser(description="HRRR Grib downloader")
    parser.add_argument("-timestamp", type=str, help="HRRR UTC date and time run to query, ie: '2021-03-30 03:15:00Z' ")
    parser.add_argument("-forecast_hour", type=int, help="HRRR forecast hour of model run to query")
    parser.add_argument(
        "-variable",
        default="PRATE",
        type=str,
        help="HRRR variable to query",
        choices=[
            "REFC",
            "RETOP",
            "VIL",
            "VIS",
            "REFD",
            "GUST",
            "UPHL",
            "UGRD",
            "VGRD",
            "PRES",
            "HGT",
            "TMP",
            "SPFH",
            "DPT",
            "WIND",
            "DSWRF",
            "VBDSF",
            "CPOFP",
            "PRATE",
            "APCP",
            "WEASD",
            "FROZR",
            "CSNOW",
            "CICEP",
            "CFRZR",
            "CRAIN",
            "TCOLWold",
            "TCOLIold",
            "ULWRF",
            "DLWRF",
            "USWRF",
            "VDDSF",
            "SBT123",
            "SBT124",
            "SBT113",
            "SBT114",
        ],
    )
    parser.add_argument(
        "-level",
        default="surface",
        type=str,
        help="level of the requested variable",
        choices=[
            "entire atmosphere",
            "cloud top",
            "surface",
            "1000 m above ground",
            "4000 m above ground",
            "5000-2000 m above ground",
            "80 m above ground",
            "2 m above ground",
            "10 m above ground",
            "cloud ceiling",
            "cloud base",
            "top of atmosphere",
        ],
    )
    parser.add_argument(
        "-out_dir", default="/tmp", type=str, help="directory to save grib files"
    )
    args = parser.parse_args()
    asyncio.run(main(args))
