from datetime import datetime, timedelta
from ftplib import FTP
from pathlib import Path
from typing import List, Tuple, TypeVar

import netCDF4
import numpy as np
import tensorflow as tf
from pvpf.property.tfrecord_property import TFRecordProperty
from pvpf.tfrecord.io import write_tfrecord
from pvpf.utils.date_range import date_range

HOST = "ftp.ptree.jaxa.jp"
USER = "simply-human1104_g.ecc.u-tokyo.ac.jp"
PASSWORD = "SP+wari8"

LONGITUDE, LATITUDE = 139.9846, 35.3277
IMAGE_SIZE = (2401, 2401)
TEMP_PATH = Path(".").joinpath("temp.nc")

NetCDFDataset = TypeVar("NetCDFDataset")


def center_index(dataset: NetCDFDataset) -> Tuple[int, int]:
    longitude = np.array(dataset.variables["longitude"])
    latitude = np.array(dataset.variables["latitude"])
    longi_index = np.argmin(np.abs(longitude - LONGITUDE))
    lati_index = np.argmin(np.abs(latitude - LATITUDE))
    return longi_index, lati_index


def crop(
    image: np.ndarray, center: Tuple[int, int], size: Tuple[int, int]
) -> np.ndarray:
    assert len(image.shape) == 2
    c_y, c_x = center
    size_x, size_y = size
    start_x = c_x - (size_x - 1) // 2
    end_x = c_x + size_x // 2
    start_y = c_y - (size_y - 1) // 2
    end_y = c_y + size_y // 2
    ans = image[start_x : end_x + 1, start_y : end_y + 1]
    return ans


def cycle_datetime(dt: datetime, size: Tuple[int, int]) -> List[np.ndarray]:
    month = dt.month
    hour = dt.hour
    month_cos = np.cos(2 * np.pi * month / 12)
    month_sin = np.sin(2 * np.pi * month / 12)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    features = [month_cos, month_sin, hour_cos, hour_sin]
    return [np.ones(size, dtype=np.float32) * value for value in features]


def to_numpy(
    dataset: NetCDFDataset,
    dt: datetime,
    center: Tuple[int, int],
    size: Tuple[int, int],
    feature_names: List[str],
) -> np.ndarray:
    ans = list()
    ans.extend(
        [
            crop(np.array(dataset.variables[key]), center, size)
            for key in feature_names
            if key in dataset.variables.keys()
        ]
    )
    return np.stack(ans, axis=-1)


def fetch_netcdf(dt: datetime) -> NetCDFDataset:
    ftp = FTP(host=HOST, user=USER, passwd=PASSWORD)
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    path = f"RETR /pub/himawari/L2/CLP/010/{year:04}{month:02}/{day:02}/{hour:02}/NC_H08_{year:04}\
{month:02}{day:02}_{hour:02}00_L2CLP010_FLDK.02401_02401.nc"
    if TEMP_PATH.exists():
        TEMP_PATH.unlink()
    with TEMP_PATH.open("wb") as f:
        ftp.retrbinary(
            path,
            f.write,
        )
    ds_nc = netCDF4.Dataset(str(TEMP_PATH))
    TEMP_PATH.unlink()
    return ds_nc


def fetch_cloud_feature(
    dt: datetime, size: Tuple[int, int], feature_names: List[str]
) -> np.ndarray:
    ds_nc = fetch_netcdf(dt)
    center = center_index(ds_nc)
    ans = to_numpy(ds_nc, dt, center, size, feature_names)
    return ans


def fetch_cloud_feature_with_linear_interpolation(
    dt: datetime, size: Tuple[int, int], feature_names: List[str]
):
    try:
        ans = fetch_cloud_feature(dt, size, feature_names)
    except Exception:
        left_step = 0
        left = None
        while left is None:
            left_step += 1
            left_dt = dt - timedelta(minutes=10 * left_step)
            try:
                left = fetch_cloud_feature(left_dt, size, feature_names)
            except Exception:
                left = None
        right_step = 0
        right = None
        while right is None:
            right_step += 1
            right_dt = dt - timedelta(minutes=10 * right_step)
            try:
                right = fetch_cloud_feature(right_dt, size, feature_names)
            except Exception:
                right = None
        ans = (right_step * left + left_step * right) / (left_step + right_step)
    dtinfo = np.stack(cycle_datetime(dt, size), axis=-1)
    ans = np.concatenate([ans, dtinfo], axis=-1)
    return ans


def create_segment_dataset(
    prop: TFRecordProperty, start: datetime, end: datetime
) -> tf.data.Dataset:
    ans = list()
    for dt in date_range(start, end, prop.time_unit):
        size = prop.image_size
        data = fetch_cloud_feature_with_linear_interpolation(
            dt, size, prop.feature_names
        )
        ans.append(data)
    ans = np.stack(ans)
    ans = tf.data.Dataset.from_tensor_slices(ans)
    return ans


def create_jaxa_tfrecord(prop: TFRecordProperty) -> Path:
    window = timedelta(days=7)
    dir_path = prop.dir_path
    feature_path = dir_path.joinpath("feature")
    for index, start in enumerate(date_range(prop.start, prop.end, window)):
        file_path = feature_path.joinpath(f"segment{index}.tfrecord")
        end = min(prop.end, start + window)
        dataset = create_segment_dataset(prop, start, end)
        write_tfrecord(file_path, dataset)
    return dir_path
