import glob
import xarray as xr

def gosat_target_file(landscan_directory, target_date_str):
    try:
        year, month = target_date_str.split('-')[:2]
        pattern = f"{landscan_directory}/gosat{year}{month}.tif"
        print(pattern)
        matches = glob.glob(pattern)
        return matches[0] if matches else None
    except Exception as e:
        print("An error occurred:", e)
        return None

def gosat_read_file(landscan_dir, overpass_date):
    gosat_file = gosat_target_file(landscan_dir, overpass_date)
    gosat_ds = xr.open_dataset(gosat_file).rename({'band_data': 'gosat'})
    gosat_ds.close()
    return gosat_ds

if __name__ == '__main__':
    landscan_dir = 'D:\\code\\python\\jcl_xco2-main\\data\\GOSAT'
    print(gosat_read_file(landscan_dir, '2016-02-01'))