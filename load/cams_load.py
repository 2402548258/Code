import glob
import xarray as xr
from load.era5_load import adjust


def get_cams_file_for_year(cams_directory, target_date_str):
    try:
        year, month = target_date_str.split('-')[:2]
        file_pattern = f"{cams_directory}/cam_{year}*.nc"
        print(f"Searching for files with pattern: {file_pattern}")
        matches = glob.glob(file_pattern)
        return matches[0] if matches else None
    except Exception as e:
        print(f"An error occurred while finding CAMS file: {e}")
        return None


def cams_read_file(cams_dir, overpass_date):
    cams_file = get_cams_file_for_year(cams_dir, overpass_date)

    if cams_file is None:
        print("No CAMS file found for the given date.")
        return None

    try:
        cams_ds = xr.open_dataset(cams_file).rename({'tcco2': 'cams'})
        cams_ds = adjust(cams_ds)
        cams_ds = cams_ds.sel(valid_time=overpass_date)
        cams_ds = cams_ds.rename({'valid_time': 'time'})
        cams_ds.close()
        return cams_ds
    except Exception as e:
        print(f"An error occurred while reading CAMS data: {e}")
        return None


if __name__ == '__main__':
    landscan_dir = 'D:\\code\\python\\jcl_xco2-main\\data\\CAMS-EGG4 XCO2'
    print(cams_read_file(landscan_dir, '2017-01-01').to_dataframe())
