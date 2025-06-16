import os
import numpy as np
import pandas as pd
import xarray as xr
from osgeo import gdal
from datetime import datetime, timedelta


def modis_target_file(modis_directory, target_date_str):
    def filename_to_date(filename):
        parts = filename.split('.')
        year = int(parts[1][1:5])
        doy = int(parts[1][5:8])
        date = datetime(year, 1, 1) + timedelta(doy - 1)
        return date

    if not os.path.isdir(modis_directory):
        return "Directory not found."

    try:
        target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
    except ValueError:
        return "Invalid target date format. Please provide date in 'YYYY-MM-DD' format."

    files_data = []
    for filename in os.listdir(modis_directory):
        if filename.endswith('.hdf'):
            file_date = filename_to_date(filename)
            file_path = os.path.join(modis_directory, filename)
            date_diff = abs((file_date - target_date).days)
            files_data.append((file_path, file_date, date_diff))

    if not files_data:
        return "No HDF files found in the directory."

    df = pd.DataFrame(files_data, columns=['FilePath', 'Date', 'DateDiff'])

    min_diff_row = df.loc[df['DateDiff'].idxmin()]
    closest_file_path = min_diff_row['FilePath']
    return closest_file_path

def define_coordinates_modis(lat_bounds, lon_bounds, resolution):
    lon = np.arange(lon_bounds[0], lon_bounds[1], resolution[1]) + 0.05
    lat = np.arange(lat_bounds[1], lat_bounds[0], -resolution[0]) + 0.05
    return lon, lat


def hdf_to_xr_modis(hdf_file):
    try:
        ndvi_subset = "HDF4_EOS:EOS_GRID:" + hdf_file + ":MODIS_Grid_16Day_VI_CMG:CMG 0.05 Deg 16 days NDVI"
        evi_subset = "HDF4_EOS:EOS_GRID:" + hdf_file + ":MODIS_Grid_16Day_VI_CMG:CMG 0.05 Deg 16 days EVI"

        ndvi_ds = gdal.Open(ndvi_subset, gdal.GA_ReadOnly)
        evi_ds = gdal.Open(evi_subset, gdal.GA_ReadOnly)
        if ndvi_ds is None or evi_ds is None:
            raise RuntimeError("Failed to open HDF dataset")

        ndvi_data = ndvi_ds.ReadAsArray()
        evi_data = evi_ds.ReadAsArray()
        if ndvi_data is None or evi_data is None:
            raise RuntimeError("Failed to read data from HDF dataset")

        ndvi_data = np.where(ndvi_data == -3000, np.nan, ndvi_data)
        evi_data = np.where(evi_data == -3000, np.nan, evi_data)
        longitude, latitude = define_coordinates_modis([-90, 90], [-180, 180], [0.05, 0.05])

        ds = xr.Dataset(
            {
                "ndvi": (["latitude", "longitude"], ndvi_data),
                "evi": (["latitude", "longitude"], evi_data),
            },
            coords={
                "longitude": longitude,
                "latitude": latitude,
            },
        )

        return ds
    except Exception as e:
        print("An error occurred:", e)
        return None

def modis_ndvi_read_file(ndvi_dir, overpass_date):
    ndvi_file = modis_target_file(ndvi_dir, overpass_date)
    print(ndvi_file)
    ndvi_ds = hdf_to_xr_modis(ndvi_file)
    return ndvi_ds


if __name__ == '__main__':
    ndvi_file = 'D:\\code\\python\\jcl_xco2-main\\data\\MODIS NDVI'
    print(modis_ndvi_read_file(ndvi_file, '2016-01-01'))



