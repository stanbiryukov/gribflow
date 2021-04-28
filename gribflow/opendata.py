import datetime


def get_models():
    models = {
        "hrrr": {
            "url": "hrrr.{timestamp.year:04d}{timestamp.month:02d}{timestamp.day:02d}/{product}/hrrr.t{timestamp.hour:02d}z.{file}f{forecast_hour:02d}.grib2",
            "base_urls": [
                "https://noaa-hrrr-bdp-pds.s3.amazonaws.com",
                "https://storage.googleapis.com/high-resolution-rapid-refresh",
            ],
            "products": {
                "conus": {
                    "files": {
                        "wrfsubh": {
                            "run_hour_delta": 1,  # difference b/w model UTC runs in hours
                            "fcst_hour_delta": 1,  # difference b/w forecast files in hours
                            "max_hour_fcst": 18,  # max forecast time out
                            "within_file_timesteps": [
                                datetime.timedelta(minutes=15),
                                datetime.timedelta(minutes=30),
                                datetime.timedelta(minutes=45),
                                datetime.timedelta(minutes=60),
                            ],
                        }
                    },
                },
            },
        },
        "gfs": {
            "url": "gfs.{timestamp.year:04d}{timestamp.month:02d}{timestamp.day:02d}/{timestamp.hour:02d}/{product}/gfs.t{timestamp.hour:02d}z.{file}.f{forecast_hour:03d}",
            "base_urls": [
                "https://noaa-gfs-bdp-pds.s3.amazonaws.com",
                "https://storage.googleapis.com/global-forecast-system",
            ],
            "products": {
                "atmos": {
                    "files": {
                        "pgrb2.0p25": {
                            "run_hour_delta": 6,
                            "fcst_hour_delta": 1,
                            "max_hour_fcst": 378,
                            "within_file_timesteps": [
                                datetime.timedelta(hours=1),
                            ],
                        },
                    },
                },
            },
        },
        "rap": {
            "url": "rap.{timestamp.year:04d}{timestamp.month:02d}{timestamp.day:02d}/rap.t{timestamp.hour:02d}z.{file}f{forecast_hour:02d}.grib2",
            "base_urls": [
                "https://noaa-rap-pds.s3.amazonaws.com",
            ],
            "products": {
                "conus": {
                    "files": {
                        "awp130pgrb": {
                            "run_hour_delta": 1,
                            "fcst_hour_delta": 1,
                            "max_hour_fcst": 21,
                            "within_file_timesteps": [
                                datetime.timedelta(hours=1),
                            ],
                        },
                    },
                },
            },
        },
    }
    return models
