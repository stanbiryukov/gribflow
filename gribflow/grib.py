import contextlib
import ctypes
import os
import struct
from ctypes.util import find_library
import datetime

import numpy as np


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
        self, libeccodes_loc=None,
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


def _read_headers(file, fgrib=FastGrib()):
    """
    Partial utility function to read headers of many grib files.
    """
    with open(file, "rb") as f:
        hdrs = fgrib.get_headers(f.read())
    return hdrs


def _read_vals(file, fgrib=FastGrib()):
    """
    Partial utility function to read values of many grib files.
    """
    with open(file, "rb") as f:
        lats, lons, vals = fgrib.get_values(f.read())
    return vals


def get_valid_time(validityDate, validityTime):
    """
    Parse valid date and time into datetime object.
    """
    validityDate = str(validityDate)
    validityTime = f"{int(validityTime):04}"
    yr = int(validityDate[:4])
    mo = int(validityDate[4:6])
    day = int(validityDate[6:8])
    hour = int(validityTime[:2])
    minute = int(validityTime[2:4])
    return datetime.datetime(yr, mo, day, hour, minute)
