import argparse
import asyncio
import base64
import contextlib
import ctypes
import datetime
import io
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import urllib
from ctypes.util import find_library
from functools import partial

import aiohttp
import numpy as np


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
    lines = response.split(':\n')
    for line in lines:
        line_ = line.split(':')
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
    matches = [x[0] for x in gribidx if (x[4] == variable) & (x[5] == level) & ( p.sub('', x[6]) == forecast)]
    return matches


def get_byte_ranges(dlocs: list, gribidx: list):
    """
    byte range is that row index's second column and ends with the next row's byte start.
    """
    return [ (gribidx[x][2], gribidx[x+1][2]) for x in dlocs]

def download_grib_chunk(url: str, path: str, _range=None):
    # print(f"Fetching: {url} with byte header: {_range} and saving to: {path}")
    req = urllib.request.Request(url, method="GET")
    if _range:
        req.add_header("Range", _range)  # 'bytes=b0-b1)'
    with urllib.request.urlopen(req) as response:
        with open(path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)


def download_files(args, idx_url: str, gribidx: list, cfg: list):
    """
    Download the files and save a unique filename based on metadata collected.
    """
    path_base = ''.join(idx_url.partition(args.model.lower())[1:]).replace('.grib2.idx','').replace('/','')
    [
        download_grib_chunk(
            url=idx_url.replace(".idx", ""),
            path = f"{args.out_dir}/{path_base}_{gribidx[x[0]][4].replace(' ', '_').strip()}_{gribidx[x[0]][5].replace(' ', '_').strip() }_{gribidx[x[0]][6].replace(' ', '_').strip() }.grib2",
            _range=f"bytes={x[1][0]}-{x[1][1]}",
        )
        for x in cfg
    ]


class grib_context(ctypes.Structure):
    pass


class grib_handle(ctypes.Structure):
    pass


class FastGrib:
    """
    Read lat, lon, and array values from a sliced grib file.

    Parameters
    ----------
    libeccodes_loc: string
        location of local libeccodes install. If None, tries to get it (Ubuntu/Debian)

    Examples
    ----------
    fgrib = FastGrib()
    with open(grib_file, "rb") as f:
        hdrs = fgrib.get_headers(f.read())
        f.seek(0)
        lats, lons, vals = fgrib.get_values(f.read())

    """

    def __init__(
        self,
        libeccodes_loc=None,
    ):
        self.libeccodes_loc = (
            self.find_eccodes() if libeccodes_loc is None else libeccodes_loc
        )

        self.eccodes = ctypes.CDLL(self.libeccodes_loc)
        # _version = eccodes.grib_get_api_version()
        # print(f'eccodes: {_version}')
        # grib_get_long
        self.grib_get_long = self.eccodes.grib_get_long
        self.grib_get_long.argtypes = [
            ctypes.POINTER(grib_handle),
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_long),
        ]
        self.grib_get_long.restype = ctypes.c_int
        # grib_get_size
        self.grib_get_size = self.eccodes.grib_get_size
        self.grib_get_size.argtypes = [
            ctypes.POINTER(grib_handle),
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.grib_get_size.restype = ctypes.c_int
        # grib_handle_new_from_message_copy
        self.grib_handle_new_from_message_copy = (
            self.eccodes.grib_handle_new_from_message_copy
        )
        self.grib_handle_new_from_message_copy.argtypes = [
            ctypes.POINTER(grib_context),
            ctypes.c_void_p,
            ctypes.c_long,
        ]
        self.grib_handle_new_from_message_copy.restype = ctypes.POINTER(grib_handle)
        # grib_handle_delete
        self.grib_handle_delete = self.eccodes.grib_handle_delete
        self.grib_handle_delete.argtypes = [ctypes.POINTER(grib_handle)]
        self.grib_handle_delete.restype = ctypes.c_long

    def find_eccodes(self):
        libloc = find_library("eccodes")
        if not isinstance(libloc, str):
            raise OSError(2, "eccodes not found")
        else:
            return libloc

    def get_key_long(self, gh, key):
        _value = ctypes.c_long(-1)
        assert self.grib_get_long(gh, key, _value) == 0
        return _value.value

    def get_headers(self, buffer):
        # redirect STDERR
        with contextlib.redirect_stderr(None):
            gh = self.grib_handle_new_from_message_copy(
                None, buffer, len(buffer)
            )  # TODO supress stderr
            # field type
            discipline = self.get_key_long(gh, b"discipline")
            category = self.get_key_long(gh, b"parameterCategory")
            number = self.get_key_long(gh, b"parameterNumber")
            # initialization (reference) date
            significance_of_reference_time = self.get_key_long(
                gh, b"significanceOfReferenceTime"
            )
            init_date = self.get_key_long(gh, b"dataDate")
            init_time = self.get_key_long(gh, b"dataTime")
            # valid date
            valid_date = self.get_key_long(gh, b"validityDate")
            valid_time = self.get_key_long(gh, b"validityTime")
            # Forecast step
            try:
                step_type = self.get_key_long(gh, b"typeOfStatisticalProcessing")
            except:
                step_type = 255
            start_step = self.get_key_long(gh, b"startStep")
            end_step = self.get_key_long(gh, b"endStep")
            # valid_range = start_step - end_step
            valid_range = end_step - start_step
            # vertical
            type_of_first_fixed_surface = self.get_key_long(
                gh, b"typeOfFirstFixedSurface"
            )
            scale_factor_of_first_fixed_surface = self.get_key_long(
                gh, b"scaleFactorOfFirstFixedSurface"
            )
            scaled_value_of_first_fixed_surface = self.get_key_long(
                gh, b"scaledValueOfFirstFixedSurface"
            )
            first_fixed_surface = (
                scaled_value_of_first_fixed_surface
                / 10 ** scale_factor_of_first_fixed_surface
                + 0
            )
            type_of_second_fixed_surface = self.get_key_long(
                gh, b"typeOfSecondFixedSurface"
            )
            scale_factor_of_second_fixed_surface = self.get_key_long(
                gh, b"scaleFactorOfSecondFixedSurface"
            )
            scaled_value_of_second_fixed_surface = self.get_key_long(
                gh, b"scaledValueOfSecondFixedSurface"
            )
            second_fixed_surface = (
                scaled_value_of_second_fixed_surface
                / 10 ** scale_factor_of_second_fixed_surface
                + 0
            )
            """
            iScansNegatively = 0;
            jScansPositively = 1;
            jPointsAreConsecutive = 0;
            alternativeRowScanning = 0;
            """
            self.grib_handle_delete(gh)
            return (
                discipline,
                category,
                number,
                init_date,
                init_time,
                valid_date,
                valid_time,
                start_step,
                end_step,
                step_type,
                valid_range,
                type_of_first_fixed_surface,
                first_fixed_surface,
                type_of_second_fixed_surface,
                second_fixed_surface,
            )

    def headers(self, buf):
        """mark start of record and end of header"""
        # GRIB2 - Section 0
        while True:
            GRIB = buf.read(4)
            if GRIB == b"":
                raise EOFError
            elif GRIB == b"GRIB":
                break
            else:
                buf.seek(-3, os.SEEK_CUR)
        record_start = buf.tell() - 4
        _567 = buf.read(3)
        edition = struct.unpack(">B", buf.read(1))[0]
        if edition == 1:
            # GRIB2 - Sections 1-2
            record_length = struct.unpack(">L", b"\x00" + _567)[0]
            for s in [1, 2]:
                nb = struct.unpack(">L", b"\x00" + buf.read(3))[0]
                buf.seek(nb - 3, os.SEEK_CUR)
            # mark end of Section 2
            header_end = buf.tell()
        else:
            # GRIB2 - Sections 1-5
            record_length = struct.unpack(">Q", buf.read(8))[0]
            for s in [1, 2, 3, 4, 5]:
                nb = struct.unpack(">L", buf.read(4))[0]
                if struct.unpack(">B", buf.read(1))[0] != s:
                    buf.seek(-5, os.SEEK_CUR)  # rewind
                else:
                    buf.seek(nb - 5, os.SEEK_CUR)
            # mark end of Section 5
            header_end = buf.tell()
        return record_start, record_length, header_end

    def get_values(self, buffer):
        gh = self.grib_handle_new_from_message_copy(None, buffer, len(buffer))
        nvalues = ctypes.c_size_t(-1)
        assert self.grib_get_size(gh, b"values", nvalues) == 0
        nx = self.get_key_long(gh, b"Nx")
        ny = self.get_key_long(gh, b"Ny")
        assert nx * ny == nvalues.value
        j_consecutive = self.get_key_long(gh, b"jPointsAreConsecutive")
        lats = np.empty(nvalues.value)
        lons = np.empty(nvalues.value)
        vals = np.empty(nvalues.value)
        lats_p = lats.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        lons_p = lons.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        vals_p = vals.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        assert self.eccodes.codes_grib_get_data(gh, lats_p, lons_p, vals_p) == 0
        lons = np.where(lons > 180.0, lons - 360.0, lons)  # make sure WGS84
        if j_consecutive:
            lats = lats.reshape(nx, ny)
            lons = lons.reshape(nx, ny)
            vals = vals.reshape(nx, ny)
        else:
            lats = lats.reshape(ny, nx)
            lons = lons.reshape(ny, nx)
            vals = vals.reshape(ny, nx)
        return lats, lons, vals


async def main(args):
    for baseurl in ['https://noaa-hrrr-bdp-pds.s3.amazonaws.com', 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/', 'https://storage.googleapis.com/high-resolution-rapid-refresh']:
        idx_url = create_grib_idx_url_path(
            baseurl=baseurl,
            timestamp=datetime.datetime.strptime(args.timestamp, '%Y-%m-%d %H:%M:%S%z'),
            forecast_hour=args.forecast_hour,
        )
        async with aiohttp.ClientSession() as session:
            r = await fetch(session, idx_url)
        if r is not None:
            break
    gribidx = read_idx(r)
    dlocs = get_byte_locs(gribidx=gribidx, variable=args.variable, level=args.level, forecast=args.forecast)
    dranges = get_byte_ranges(dlocs=dlocs, gribidx=gribidx)
    download_files(args=args, idx_url=idx_url, gribidx=gribidx, cfg=list(zip(dlocs, dranges)))


if __name__ == "__main__":
    """
    Fetch subsets of NOAA model outputs.

    Examples
    ----------
        python get_gribs.py -timestamp '2021-03-30 03:15:00Z' -model 'HRRR' -forecast_hour 6 -variable 'PRATE' -level 'surface' -forecast 'min fcst' -out_dir '/tmp'
    """
    parser = argparse.ArgumentParser(description="Grib downloader")
    parser.add_argument(
        "-timestamp",
        type=str,
        help="UTC date and time model run to query, ie: '2021-03-30 03:00:00Z' ",
    )
    parser.add_argument(
        "-model", type=str, help="NOAA model to query"
    )
    parser.add_argument(
        "-forecast_hour", type=int, help="forecast hour of model run to query"
    )
    parser.add_argument(
        "-variable",
        default="PRATE",
        type=str,
        help="variable to query",
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
        "-forecast",
        default="min fcst",
        type=str,
        help="forecast type of the requested variable",
        choices=[
            "min acc fcst",
            "min ave fcst",
            "min fcst",
        ],
    )
    parser.add_argument(
        "-out_dir", default="/tmp", type=str, help="directory to save grib files"
    )
    args = parser.parse_args()
    asyncio.run(main(args))
