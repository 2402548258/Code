
import glob
import xarray as xr

def era_target_file(era_directory, target_date_str):
    year,month = target_date_str.split('-')[:2]
    pattern = f"{era_directory}/era_{year}*.nc"
    print(pattern)
    matches = glob.glob(pattern)
    return matches[0] if matches else None

def adjust(ds):
    lon = ds['longitude']
    new_lon = xr.where(lon > 180, lon - 360, lon)
    ds = ds.assign_coords(longitude=new_lon)
    ds = ds.sortby('longitude')
    return ds

def era_read_file(era_dir, overpass_date):
    year, month = overpass_date.split('-')[:2]
    era_file = era_target_file(era_dir, overpass_date)
    if era_file is None:
        raise FileNotFoundError(f"ERA file for date {overpass_date} not found.")
    try:
        with xr.open_dataset(era_file, engine='netcdf4') as ds_monthly:
            ds_monthly = adjust(ds_monthly)
            era_ds = ds_monthly.sel(valid_time=f"{year}-{month}")
            return era_ds.load()
    except Exception as e:
        raise RuntimeError(f"Error occurred while reading ERA file {era_file}: {e}")

if __name__ == '__main__':
    landscan_dir = 'D:\\code\python\\jcl_xco2-main\\data\\ERA-5 hourly wind vector'
    print(era_read_file(landscan_dir, '2017-01-01'))


