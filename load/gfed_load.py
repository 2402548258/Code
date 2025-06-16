import os, glob
import numpy as np
import xarray as xr
from osgeo import gdal

def gfed_target_file(gfed_directory, target_date_str):
    try:
        year, month = target_date_str.split('-')[:2]
        pattern = f"{gfed_directory}/GFED4.1s_{year}*.hdf5"
        matches = glob.glob(pattern)
        return matches[0] if matches else None
    except Exception as e:
        print("An error occurred:", e)
        return None

def adjust_coordinates(lat_bounds, lon_bounds, resolution):
    lon = np.arange(lon_bounds[0], lon_bounds[1], resolution[1]) + 0.25
    lat = np.arange(lat_bounds[1], lat_bounds[0], -resolution[0]) + 0.25
    return lon, lat

def hdf_to_xr_gfed(hdf_file, month):
    try:
        gfed_subset = "HDF5:" + hdf_file + "://emissions/" + str(month) + "/C"
        gfed_ds = gdal.Open(gfed_subset, gdal.GA_ReadOnly)
        if gfed_ds is None:
            raise RuntimeError("Failed to open HDF dataset")
        gfed_data = gfed_ds.ReadAsArray()
        if gfed_data is None:
            raise RuntimeError("Failed to read data from HDF dataset")
        gfed_data = np.where(gfed_data == -3000, np.nan, gfed_data)
        longitude, latitude = adjust_coordinates([-90, 90], [-180, 180], [0.25, 0.25])
        ds = xr.Dataset(
            {
                "gfed": (["latitude", "longitude"], gfed_data),
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

def gfed_read_file(gfed_dir, overpass_date):
    gfed_file = gfed_target_file(gfed_dir, overpass_date)
    gfed_ds = hdf_to_xr_gfed(gfed_file, overpass_date[5:7])
    gfed_ds.close()
    return gfed_ds


if __name__ == '__main__':
    landscan_dir = 'D:\\code\\python\\jcl_xco2-main\\data\\GFED emissions'
    print(gfed_read_file(landscan_dir, '2016-01-01'))