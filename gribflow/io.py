import argparse
import asyncio
import base64
import datetime
import io
import json
import shutil
import urllib
from functools import partial
from typing import List

import aiohttp
import blosc
import numpy as np
import pandas as pd
import subprocess
import ctypes
import os
import struct
import sys


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
    return [(dparsed.loc[r][1], dparsed.loc[r + 1][1]) for r in dlocs.index]


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
            path=f"{out_dir}/{'_'.join(idx_url.rsplit('/')[-3:]).replace('.grib2.idx', '')}_{x[0][0].strip()}_{''.join(x[0][1]).replace(' ','_').strip()}_{''.join(x[0][2]).replace(' ','_').strip()}.grib2",
            _range=f"bytes={x[1][0]}-{x[1][1]}",
        )
        for x in cfg
    ]


def encode_array(array, compressor=partial(blosc.pack_array, cname="lz4")):
    """
    Compressor numpy array to json-safe string
    """
    cdata = base64.urlsafe_b64encode(compressor(array)).decode("utf-8")
    return cdata


def decode_array(cdata, compressor=blosc.unpack_array):
    """
    Decode string to bytes and uncompress to numpy array
    """
    data = compressor(base64.urlsafe_b64decode(cdata))
    return data


class grib_context(ctypes.Structure):
    pass


class grib_handle(ctypes.Structure):
    pass


class FastGrib:
    '''
    Read lat, lon, and array values from a sliced grib file.

    Parameters
    ----------
    libeccodes_loc: string
        location of local libeccodes install. If None, tries to get it (Ubuntu/Debian)

    Examples
    ----------
    fgrib = FastGrib()
    with open(grib_file, "rb") as f:
        lats, lons, vals = fgrib.get_values(f.read())

    '''

    def __init__(
        self,
        libeccodes_loc=None,
    ):
        self.libeccodes_loc = self.get_libeccodes() if libeccodes_loc is None else libeccodes_loc

        eccodes = ctypes.CDLL(self.libeccodes_loc)
        # _version = eccodes.grib_get_api_version()
        # print(f'eccodes: {_version}')
        # grib_get_long
        grib_get_long = eccodes.grib_get_long
        grib_get_long.argtypes = [ctypes.POINTER(grib_handle), ctypes.c_char_p, ctypes.POINTER(ctypes.c_long)]
        grib_get_long.restype = ctypes.c_int
        # grib_get_size
        grib_get_size = eccodes.grib_get_size
        grib_get_size.argtypes = [ctypes.POINTER(grib_handle), ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t)]
        grib_get_size.restype = ctypes.c_int
        # grib_handle_new_from_message_copy
        grib_handle_new_from_message_copy = eccodes.grib_handle_new_from_message_copy
        grib_handle_new_from_message_copy.argtypes = [ctypes.POINTER(grib_context), ctypes.c_void_p, ctypes.c_long]
        grib_handle_new_from_message_copy.restype = ctypes.POINTER(grib_handle)
        # grib_handle_delete
        grib_handle_delete = eccodes.grib_handle_delete
        grib_handle_delete.argtypes = [ctypes.POINTER(grib_handle)]
        grib_handle_delete.restype = ctypes.c_long

    def get_libeccodes(self):
        try:
            libloc = subprocess.check_output(['bash', '-c', 'dpkg -L libeccodes-dev'])
        except Exception as e:
            raise OSError(2, "libeccodes-dev not found")
        libloc = [x for x in libloc.decode('utf-8').splitlines() if 'libeccodes.so' in x]
        return libloc[0]

    def get_key_long(self, gh, key):
        _value = ctypes.c_long(-1)
        assert grib_get_long(gh, key, _value) == 0
        return _value.value

    def get_headers(self, buffer):
        # redirect STDERR
        fd = sys.stderr.fileno()
        with os.fdopen(os.dup(fd), "w") as _stderr, open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), fd)
            gh = grib_handle_new_from_message_copy(None, buffer, len(buffer))   # TODO supress stderr
            # field type
            discipline = self.get_key_long(gh, b"discipline")
            category   = self.get_key_long(gh, b"parameterCategory")
            number     = self.get_key_long(gh, b"parameterNumber")
            # initialization (reference) date
            significance_of_reference_time = self.get_key_long(gh, b"significanceOfReferenceTime")
            init_date                      = self.get_key_long(gh, b"dataDate")
            init_time                      = self.get_key_long(gh, b"dataTime")
            # valid date
            valid_date                  = self.get_key_long(gh, b"validityDate")
            valid_time                  = self.get_key_long(gh, b"validityTime")
            # Forecast step
            try:
                step_type = self.get_key_long(gh, b"typeOfStatisticalProcessing")
            except:
                step_type = 255
            start_step = self.get_key_long(gh, b"startStep")
            end_step   = self.get_key_long(gh, b"endStep")
            #valid_range = start_step - end_step
            valid_range = end_step - start_step
            # vertical
            type_of_first_fixed_surface          = self.get_key_long(gh, b"typeOfFirstFixedSurface")
            scale_factor_of_first_fixed_surface  = self.get_key_long(gh, b"scaleFactorOfFirstFixedSurface")
            scaled_value_of_first_fixed_surface  = self.get_key_long(gh, b"scaledValueOfFirstFixedSurface")
            first_fixed_surface  = scaled_value_of_first_fixed_surface / 10**scale_factor_of_first_fixed_surface + 0
            type_of_second_fixed_surface         = self.get_key_long(gh, b"typeOfSecondFixedSurface")
            scale_factor_of_second_fixed_surface = self.get_key_long(gh, b"scaleFactorOfSecondFixedSurface")
            scaled_value_of_second_fixed_surface = self.get_key_long(gh, b"scaledValueOfSecondFixedSurface")
            second_fixed_surface  = scaled_value_of_second_fixed_surface / 10**scale_factor_of_second_fixed_surface + 0
            """
            iScansNegatively = 0;
            jScansPositively = 1;
            jPointsAreConsecutive = 0;
            alternativeRowScanning = 0;
            """
            grib_handle_delete(gh)
            # restore
            os.dup2(_stderr.fileno(), fd)
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
                second_fixed_surface
            )

    def headers(self, buf):
        """ mark start of record and end of header
        """
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
            for s in [1,2]:
                nb = struct.unpack('>L', b"\x00" + buf.read(3))[0]
                buf.seek(nb-3, os.SEEK_CUR)
            # mark end of Section 2
            header_end = buf.tell()
        else:
            # GRIB2 - Sections 1-5
            record_length = struct.unpack('>Q', buf.read(8))[0]
            for s in [1,2,3,4,5]:
                nb = struct.unpack('>L', buf.read(4))[0]
                if struct.unpack(">B", buf.read(1))[0] != s:
                    buf.seek(-5, os.SEEK_CUR) # rewind
                else:
                    buf.seek(nb-5, os.SEEK_CUR)
            # mark end of Section 5
            header_end = buf.tell()
        return record_start, record_length, header_end

    def get_values(self, buffer):
        gh = grib_handle_new_from_message_copy(None, buffer, len(buffer))
        nvalues = ctypes.c_size_t(-1)
        assert grib_get_size(gh, b"values", nvalues) == 0
        nx = self.get_key_long(gh, b"Nx")
        ny = self.get_key_long(gh, b"Ny")
        assert nx*ny == nvalues.value
        j_consecutive = self.get_key_long(gh, b"jPointsAreConsecutive")
        lats = np.empty(nvalues.value)
        lons = np.empty(nvalues.value)
        vals = np.empty(nvalues.value)
        lats_p = lats.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        lons_p = lons.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        vals_p = vals.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        assert eccodes.codes_grib_get_data(gh, lats_p, lons_p, vals_p) == 0
        lons = np.where(lons > 180., lons-360., lons) # make sure WGS84
        if j_consecutive:
            lats = lats.reshape(nx,ny)
            lons = lons.reshape(nx,ny)
            vals = vals.reshape(nx,ny)
        else:
            lats = lats.reshape(ny,nx)
            lons = lons.reshape(ny,nx)
            vals = vals.reshape(ny,nx)
        return lats, lons, vals


async def main(args):
    idx_url = create_grib_idx_url_path(
        timestamp=pd.to_datetime(args.timestamp, utc=True),
        forecast_hour=args.forecast_hour,
    )
    async with aiohttp.ClientSession() as session:
        r = await fetch(session, idx_url)

    dparsed = parse_to_df(r)
    dlocs = get_byte_locs(dparsed, variable=args.variable, level=args.level)
    dmeta = get_forecast_metadata(dlocs)
    dranges = get_byte_ranges(dlocs=dlocs, dparsed=dparsed)
    download_files(idx_url=idx_url, out_dir=args.out_dir, cfg=list(zip(dmeta, dranges)))


if __name__ == "__main__":
    """
    Fetch subsets of HRRR sub-hourly forecast outputs.
    
    Examples
    ----------
        python get_gribs.py -timestamp '2021-03-30 03:15:00Z' -forecast_hour 6 -variable 'PRATE' -level 'surface' -out_dir '/tmp'
    """
    parser = argparse.ArgumentParser(description="HRRR Grib downloader")
    parser.add_argument(
        "-timestamp",
        type=str,
        help="HRRR UTC date and time run to query, ie: '2021-03-30 03:15:00Z' ",
    )
    parser.add_argument(
        "-forecast_hour", type=int, help="HRRR forecast hour of model run to query"
    )
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
