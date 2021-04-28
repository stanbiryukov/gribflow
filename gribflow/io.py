import argparse
import asyncio
import datetime
import re

import aiohttp

from gribflow.opendata import get_models


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


def create_grib_idx_url_path(
    baseurl: str,
    url: str,
    product: str,
    file: str,
    timestamp: datetime,
    forecast_hour: int,
):
    """
    Given a date and forecast hour return the expected grib.idx filepath
    """
    url = url.format(
        timestamp=timestamp, product=product, file=file, forecast_hour=forecast_hour
    )
    # print(f"{baseurl}/{url}.idx")
    return f"{baseurl}/{url}.idx"


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
        if (x[4].lower() == variable.lower())
        & (x[5].lower() == level.lower())
        & (p.sub("", x[6]).replace(" ", "").lower() == forecast.lower())
    ]
    return matches


def get_byte_ranges(dlocs: list, gribidx: list):
    """
    byte range is that row index's second column and ends with the next row's byte start.
    """
    return [(gribidx[x][2], gribidx[x + 1][2]) for x in dlocs]


async def make_request(url, path, _range):
    header = {"Range": f"bytes={_range}"}
    # print(f"Fetching {url} to {path}")
    async with aiohttp.ClientSession(headers=header) as session:
        async with session.get(url=url) as resp:
            with open(path, "wb") as f:
                async for chunk in resp.content.iter_chunked(4096):
                    f.write(chunk)
    return path


async def download_files(
    model: str, out_dir: str, idx_url: str, gribidx: list, cfg: list
):
    """
    Create aiohttp requests for all the grib chunks.
    """
    path_base = (
        "".join(idx_url.partition(model.lower())[1:])
        .replace("/", "")
        .replace("idx", "")
        .split(".grib", 1)[0]
    )
    tasks = [
        asyncio.create_task(
            make_request(
                url=idx_url.replace(".idx", ""),
                path=f"{out_dir}/{path_base}_{gribidx[x[0]][4].replace(' ', '_').strip()}_{gribidx[x[0]][5].replace(' ', '_').strip() }_{gribidx[x[0]][6].replace(' ', '_').strip() }.grib2",
                _range=f"{x[1][0]}-{x[1][1]}",
            )
        )
        for x in cfg
    ]
    results = await asyncio.gather(*tasks)
    return results


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts list in human order
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]


async def get_gribs(
    timestamp: datetime,
    forecast_hour: int,
    model: str,
    product: str,
    file: str,
    variable: str,
    level: str,
    forecast: str,
    out_dir: str,
):
    cfg = get_models()[model.lower()]
    for baseurl in cfg["base_urls"]:
        idx_url = create_grib_idx_url_path(
            baseurl=baseurl,
            url=cfg["url"],
            product=product.lower(),
            file=file.lower(),
            timestamp=datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S"),
            forecast_hour=forecast_hour,
        )
        async with aiohttp.ClientSession() as session:
            r = await fetch(session, idx_url)
        if r is not None:
            break

    if r is None:
        raise aiohttp.ClientConnectionError()

    gribidx = read_idx(r)
    dlocs = get_byte_locs(
        gribidx=gribidx,
        variable=variable,
        level=level,
        forecast=forecast,
    )
    dranges = get_byte_ranges(dlocs=dlocs, gribidx=gribidx)
    results = await download_files(
        model=model,
        out_dir=out_dir,
        idx_url=idx_url,
        gribidx=gribidx,
        cfg=list(zip(dlocs, dranges)),
    )
    return results


if __name__ == "__main__":
    """
    Fetch subsets of NOAA model outputs.

    Examples
    ----------
        python3 io.py -model 'hrrr' -product 'conus' -file 'wrfsubh' -timestamp '2021-04-11 11:00:00' -forecast_hour 1 -variable 'PRATE' -level 'surface' -forecast 'minfcst' -out_dir '/tmp'
        python3 io.py -model 'gfs' -product 'atmos' -file 'pgrb2.0p25' -timestamp '2021-04-11 12:00:00' -forecast_hour 6 -variable 'TMP' -level 'surface' -forecast 'hourfcst' -out_dir '/tmp'
    """
    parser = argparse.ArgumentParser(description="Grib downloader")
    parser.add_argument(
        "-timestamp",
        type=str,
        required=True,
        help="UTC date and time model run to query, ie: '2021-03-30 03:00:00' ",
    )
    parser.add_argument(
        "-model", type=str, default="hrrr", required=True, help="NOAA model to query"
    )
    parser.add_argument(
        "-product",
        type=str,
        default="conus",
        required=True,
        help="NOAA model product to query. Ex: 'conus', 'atmos'",
    )
    parser.add_argument(
        "-file",
        type=str,
        default="conus",
        required=True,
        help="NOAA model product file to query. Ex: 'wrfsubhf', 'pgrb2.0p25'",
    )
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
    asyncio.run(
        get_gribs(
            timestamp=args.timestamp,
            forecast_hour=args.forecast_hour,
            model=args.model,
            product=args.product,
            file=args.file,
            variable=args.variable,
            level=args.level,
            forecast=args.forecast,
            out_dir=args.out_dir,
        )
    )
