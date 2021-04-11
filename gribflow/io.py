import argparse
import asyncio
import datetime
import re

import aiofiles
import aiohttp


async def fetch(session, url):
    """
    Create aiohttp request.
    """
    async with session.get(url) as response:
        if response.status != 200:
            print(f"Missing {url}")
            return None
        elif response.status == 200:
            return await response.text()


def create_grib_idx_url_path(baseurl: str, timestamp: datetime, forecast_hour: int):
    """
    Given a date and forecast hour return the expected grib.idx filepath on the open data S3 bucket for HRRR `wrfsubh` file.
    """
    return f"{baseurl}/hrrr.{timestamp.year:04d}{timestamp.month:02d}{timestamp.day:02d}/conus/hrrr.t{timestamp.hour:02d}z.wrfsubhf{str(forecast_hour).zfill(2)}.grib2.idx"


def read_idx(response):
    """
    Read grib index file as list and insert row number into beginning
    """
    i = 0
    data = []
    lines = response.split(":\n")
    for line in lines:
        line_ = line.split(":")
        if len(line_) > 1:
            line_.insert(0, i)
            data.append(line_)
        i += 1
    return data


def get_byte_locs(gribidx: list, variable: str, level: str, forecast: str):
    """
    var is in 4th column, 5th is level, 6th is forecast type (strip up until first word)
    first column is the 0..N index we created.
    """
    p = re.compile(r"(^[^A-Za-z]+)")
    matches = [
        x[0]
        for x in gribidx
        if (x[4] == variable) & (x[5] == level) & (p.sub("", x[6]) == forecast)
    ]
    return matches


def get_byte_ranges(dlocs: list, gribidx: list):
    """
    byte range is that row index's second column and ends with the next row's byte start.
    """
    return [(gribidx[x][2], gribidx[x + 1][2]) for x in dlocs]


async def make_request(url, path, _range):
    header = {"Range": f"bytes={_range}"}
    async with aiohttp.ClientSession(headers=header) as session:
        async with session.get(url=url) as resp:
            f = await aiofiles.open(path, mode="wb")
            await f.write(await resp.read())
            await f.close()


async def download_files(args, idx_url: str, gribidx: list, cfg: list):
    """
    Create aiohttp requests for all the grib chunks.
    """
    path_base = (
        "".join(idx_url.partition(args.model.lower())[1:])
        .replace(".grib2.idx", "")
        .replace("/", "")
    )
    tasks = []
    for x in cfg:
        task = asyncio.create_task(
            make_request(
                url=idx_url.replace(".idx", ""),
                path=f"{args.out_dir}/{path_base}_{gribidx[x[0]][4].replace(' ', '_').strip()}_{gribidx[x[0]][5].replace(' ', '_').strip() }_{gribidx[x[0]][6].replace(' ', '_').strip() }.grib2",
                _range=f"{x[1][0]}-{x[1][1]}",
            )
        )
        tasks.append(task)
    results = await asyncio.gather(*tasks)
    return results


def get_models():
    models = {
        "hrrr": {
            "products": {
                "wrfsubhf": {
                    "time_delta": "1 hour",
                }
            }
        },
        "gfs": {"products": {"atmos": {"time_delta": "6 hours"}}},
    }
    return models


async def main(args):
    for baseurl in [
        "https://noaa-hrrr-bdp-pds.s3.amazonaws.com",
        "https://storage.googleapis.com/high-resolution-rapid-refresh",
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod",
    ]:
        idx_url = create_grib_idx_url_path(
            baseurl=baseurl,
            timestamp=datetime.datetime.strptime(args.timestamp, "%Y-%m-%d %H:%M:%S%z"),
            forecast_hour=args.forecast_hour,
        )
        async with aiohttp.ClientSession() as session:
            r = await fetch(session, idx_url)
        if r is not None:
            break
    gribidx = read_idx(r)
    dlocs = get_byte_locs(
        gribidx=gribidx,
        variable=args.variable,
        level=args.level,
        forecast=args.forecast,
    )
    dranges = get_byte_ranges(dlocs=dlocs, gribidx=gribidx)
    await download_files(
        args=args, idx_url=idx_url, gribidx=gribidx, cfg=list(zip(dlocs, dranges))
    )


if __name__ == "__main__":
    """
    Fetch subsets of NOAA model outputs.

    Examples
    ----------
        python get_gribs.py -model 'HRRR' -timestamp '2021-04-01 03:00:00Z' -forecast_hour 3 -variable 'PRATE' -level 'surface' -forecast 'min fcst' -out_dir '/tmp'
    """
    parser = argparse.ArgumentParser(description="Grib downloader")
    parser.add_argument(
        "-timestamp",
        type=str,
        required=True,
        help="UTC date and time model run to query, ie: '2021-03-30 03:00:00Z' ",
    )
    parser.add_argument("-model", type=str, required=True, help="NOAA model to query")
    parser.add_argument(
        "-forecast_hour",
        type=int,
        required=True,
        help="forecast hour of model run to query",
    )
    parser.add_argument(
        "-variable",
        default="PRATE",
        type=str,
        required=True,
        help="variable to query",
    )
    parser.add_argument(
        "-level",
        default="surface",
        type=str,
        required=True,
        help="level of the requested variable",
    )
    parser.add_argument(
        "-forecast",
        default="min fcst",
        type=str,
        required=True,
        help="forecast type of the requested variable",
    )
    parser.add_argument(
        "-out_dir",
        default="/tmp",
        type=str,
        required=True,
        help="directory to save grib files",
    )
    args = parser.parse_args()
    asyncio.run(main(args))
